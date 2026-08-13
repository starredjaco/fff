use heed::{Env, EnvOpenOptions};
use std::collections::HashMap;
use std::fs;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard, PoisonError, Weak};
use std::thread;
use std::time::Duration;

use super::lmdb::DbHealth;
use crate::error::{Error, Result};

pub(crate) struct EnvSpec {
    pub label: &'static str,
    pub map_size: usize,
    pub max_dbs: u32,
    pub size_cap_bytes: u64,
}

pub(crate) struct PooledEnv {
    env: Env,
    key: PathBuf,
    label: &'static str,
    map_size: usize,
    max_dbs: u32,
    health: DbHealth,
    gc_started: AtomicBool,
    dbi_lock: Mutex<()>,
}

impl Drop for PooledEnv {
    fn drop(&mut self) {
        let mut pool = lock_pool();
        // Only remove a dead entry: begin_exclusive_destroy may have removed ours.
        if pool.get(&self.key).is_some_and(|w| w.strong_count() == 0) {
            pool.remove(&self.key);
        }
        // heed closes the env right after this body; a concurrent reopen of the
        // same path rides out that gap via env_closing_event in get_or_open.
    }
}

// Cloneable handle to a process-shared LMDB env, derefs to `heed::Env`.
#[derive(Clone)]
pub(crate) struct SharedEnv(Arc<PooledEnv>);

impl Deref for SharedEnv {
    type Target = Env;
    fn deref(&self) -> &Env {
        &self.0.env
    }
}

impl std::fmt::Debug for SharedEnv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("SharedEnv").field(&self.0.env).finish()
    }
}

impl SharedEnv {
    pub(crate) fn health(&self) -> &DbHealth {
        &self.0.health
    }

    // First caller wins: GC runs once per opened env, not once per tracker.
    pub(crate) fn try_start_gc(&self) -> bool {
        !self.0.gc_started.swap(true, Ordering::AcqRel)
    }

    // LMDB forbids mdb_dbi_open from concurrent txns in the same process.
    pub(crate) fn lock_dbi_open(&self) -> MutexGuard<'_, ()> {
        self.0
            .dbi_lock
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }
}

static POOL: LazyLock<Mutex<HashMap<PathBuf, Weak<PooledEnv>>>> = LazyLock::new(Mutex::default);

const CLOSE_WAIT: Duration = Duration::from_millis(100);
const MAX_CLOSE_WAITS: u32 = 100;
const TRANSIENT_RETRY_SLEEP: Duration = Duration::from_millis(50);
const MAX_TRANSIENT_RETRIES: u32 = 8;

/// One process must never hold two LMDB envs over one path (POSIX lock rules),
/// so opens of an already-pooled canonical path return the shared handle.
pub(crate) fn get_or_open(db_path: &Path, spec: &EnvSpec) -> Result<SharedEnv> {
    fs::create_dir_all(db_path).map_err(Error::CreateDir)?;
    // Canonicalized so every spelling of a path shares one env, matching the
    // key heed keeps in its own OPENED_ENV registry.
    let key = fs::canonicalize(db_path).map_err(|e| Error::EnvOpen {
        db: spec.label,
        source: heed::Error::Io(e),
    })?;

    let mut close_waits = 0u32;
    let mut transient_retries = 0u32;
    loop {
        let mut mid_close = false;
        {
            let mut pool = lock_pool();
            if let Some(existing) = pool.get(&key).and_then(Weak::upgrade) {
                // Release the map lock before this Arc could drop re-entrantly.
                drop(pool);
                if existing.label != spec.label
                    || existing.map_size != spec.map_size
                    || existing.max_dbs != spec.max_dbs
                {
                    return Err(Error::EnvSpecMismatch {
                        path: key,
                        open_as: existing.label,
                        requested_as: spec.label,
                    });
                }
                return Ok(SharedEnv(existing));
            }

            erase_if_oversized(&key, spec);
            let result = unsafe {
                let mut opts = EnvOpenOptions::new();
                opts.map_size(spec.map_size);
                if spec.max_dbs > 0 {
                    opts.max_dbs(spec.max_dbs);
                }
                opts.open(&key)
            };

            match result {
                Ok(env) => {
                    let entry = Arc::new(PooledEnv {
                        env,
                        key: key.clone(),
                        label: spec.label,
                        map_size: spec.map_size,
                        max_dbs: spec.max_dbs,
                        health: DbHealth::new(),
                        gc_started: AtomicBool::new(false),
                        dbi_lock: Mutex::new(()),
                    });
                    pool.insert(key.clone(), Arc::downgrade(&entry));
                    drop(pool);
                    let shared = SharedEnv(entry);
                    reclaim_stale_readers(&shared, spec.label);
                    return Ok(shared);
                }
                // Same canonical path is mid-close on another thread: wait for
                // heed to signal the real close, then retry.
                Err(heed::Error::EnvAlreadyOpened) => mid_close = true,
                Err(e)
                    if is_transient_env_open_error(&e)
                        && transient_retries < MAX_TRANSIENT_RETRIES =>
                {
                    transient_retries += 1;
                    tracing::debug!(
                        path = %key.display(),
                        transient_retries,
                        error = ?e,
                        "transient LMDB env open error, retrying"
                    );
                }
                Err(e) => {
                    return Err(Error::EnvOpen {
                        db: spec.label,
                        source: e,
                    });
                }
            }
        }

        if mid_close {
            close_waits += 1;
            if close_waits > MAX_CLOSE_WAITS {
                return Err(Error::EnvOpen {
                    db: spec.label,
                    source: heed::Error::EnvAlreadyOpened,
                });
            }
            match heed::env_closing_event(&key) {
                Some(event) => {
                    event.wait_timeout(CLOSE_WAIT);
                }
                // Key spelling can differ from heed's on Windows: plain backoff.
                None => thread::sleep(Duration::from_millis(2)),
            }
        } else {
            thread::sleep(TRANSIENT_RETRY_SLEEP);
        }
    }
}

/// Refuses when other trackers share the env. On success the entry is unpooled;
/// the caller drops its last handle, waits on the event, then deletes the files.
pub(crate) fn begin_exclusive_destroy(shared: &SharedEnv) -> Result<Option<heed::EnvClosingEvent>> {
    let mut pool = lock_pool();
    let holders = Arc::strong_count(&shared.0);
    if holders > 1 {
        return Err(Error::DbInUse {
            db: shared.0.label,
            path: shared.0.key.clone(),
            holders: holders - 1,
        });
    }
    pool.remove(&shared.0.key);
    Ok(heed::env_closing_event(&shared.0.key))
}

fn lock_pool() -> MutexGuard<'static, HashMap<PathBuf, Weak<PooledEnv>>> {
    // The map only holds Weak refs, so a poisoned lock is still consistent.
    POOL.lock().unwrap_or_else(PoisonError::into_inner)
}

// Concurrent mdb_env_open calls on the same path can race on macOS
// this is for some reason fixabtly by simple retry of the open
fn is_transient_env_open_error(err: &heed::Error) -> bool {
    match err {
        heed::Error::Io(io) => matches!(
            io.kind(),
            std::io::ErrorKind::InvalidInput | std::io::ErrorKind::NotFound
        ),
        _ => false,
    }
}

// Reclaim reader slots left by processes that died without cleanup. Must run
// before the first read txn, or a fresh env can hit MDB_READERS_FULL.
fn reclaim_stale_readers(env: &Env, db: &'static str) {
    match env.clear_stale_readers() {
        Ok(cleared) if cleared > 0 => {
            tracing::warn!(cleared, db, "reclaimed stale LMDB reader slots at open");
        }
        Ok(_) => {}
        Err(e) => tracing::debug!("clear_stale_readers at open failed: {e}"),
    }
}

fn erase_if_oversized(db_path: &Path, spec: &EnvSpec) {
    let data = db_path.join("data.mdb");
    let Ok(meta) = fs::metadata(&data) else {
        return;
    };
    if meta.len() <= spec.size_cap_bytes {
        return;
    }

    tracing::error!(
        path = %db_path.display(),
        size = meta.len(),
        cap = spec.size_cap_bytes,
        "LMDB db exceeds size cap, erasing"
    );
    let _ = fs::remove_file(&data);
    let _ = fs::remove_file(db_path.join("lock.mdb"));
}
