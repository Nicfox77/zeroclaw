//! Per-turn inactivity supervision. This owner creates the last-progress and
//! bounded-wait facts; adapters retain only scoped observation handles.
use parking_lot::Mutex;
use std::{
    collections::HashMap,
    future::Future,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};
use tokio::{sync::watch, time::Instant};
use tokio_util::sync::CancellationToken;
use zeroclaw_api::tool::{ToolActivity, ToolActivityLease};

tokio::task_local! { static CURRENT: Option<Activity>; }

#[derive(Debug)]
struct State {
    last_progress: Instant,
    waits: HashMap<u64, Instant>,
}
#[derive(Debug)]
struct Inner {
    state: Mutex<State>,
    changed: watch::Sender<()>,
    next_wait: AtomicU64,
}
#[derive(Debug, Clone)]
pub struct Activity(Arc<Inner>);

#[derive(Debug, thiserror::Error)]
#[error("turn inactive for {seconds}s")]
pub struct IdleTimeout {
    pub seconds: u64,
}

impl Activity {
    pub(crate) fn new() -> Self {
        let (changed, _) = watch::channel(());
        Self(Arc::new(Inner {
            state: Mutex::new(State {
                last_progress: Instant::now(),
                waits: HashMap::new(),
            }),
            changed,
            next_wait: AtomicU64::new(0),
        }))
    }

    /// A foreground waiter observes the job's canonical activity. No timer or
    /// polling heartbeat manufactures progress, and no policy is copied here.
    pub(crate) async fn observe<T>(&self, future: impl Future<Output = T>) -> T {
        let Some(parent) = Self::current() else {
            return future.await;
        };
        if Arc::ptr_eq(&self.0, &parent.0) {
            return future.await;
        }
        let mut updates = self.0.changed.subscribe();
        let mut observed = self.0.state.lock().last_progress;
        tokio::pin!(future);
        loop {
            tokio::select! {
                biased;
                result = &mut future => return result,
                _ = updates.changed() => {
                    let latest = self.0.state.lock().last_progress;
                    if latest > observed {
                        parent.progress();
                        observed = latest;
                    }
                }
            }
        }
    }

    pub fn current() -> Option<Self> {
        CURRENT.try_with(Clone::clone).ok().flatten()
    }
    pub fn progress_current() {
        if let Some(activity) = Self::current() {
            activity.progress();
        }
    }
    pub fn tool_observer() -> Option<Arc<dyn ToolActivity>> {
        Self::current().map(|activity| Arc::new(activity) as Arc<dyn ToolActivity>)
    }
    fn deadline(&self, idle: Duration) -> Instant {
        let state = self.0.state.lock();
        // Configuration validation rejects durations Instant cannot represent.
        let idle_deadline = state
            .last_progress
            .checked_add(idle)
            .unwrap_or(state.last_progress);
        state
            .waits
            .values()
            .copied()
            .fold(idle_deadline, Instant::max)
    }
}
impl ToolActivity for Activity {
    fn progress(&self) {
        self.0.state.lock().last_progress = Instant::now();
        self.0.changed.send_replace(());
    }
    fn bounded_wait(&self, timeout: Duration) -> Box<dyn ToolActivityLease> {
        // IDs are only transient lease identities. A turn cannot create 2^64
        // simultaneously live leases; no policy is copied into this owner.
        let id = self.0.next_wait.fetch_add(1, Ordering::Relaxed);
        if let Some(deadline) = timeout
            .checked_add(Duration::from_secs(5))
            .and_then(|budget| Instant::now().checked_add(budget))
        {
            self.0.state.lock().waits.insert(id, deadline);
            self.0.changed.send_replace(());
        }
        Box::new(WaitLease {
            activity: self.clone(),
            id,
        })
    }
}
struct WaitLease {
    activity: Activity,
    id: u64,
}
impl ToolActivityLease for WaitLease {}
impl Drop for WaitLease {
    fn drop(&mut self) {
        self.activity.0.state.lock().waits.remove(&self.id);
        self.activity.progress();
    }
}

/// No total-duration cap. Real progress refreshes the deadline; bounded quiet
/// operations borrow their existing execution budget. No background task owns
/// the engine, so dropping this future also drops supervision and all work.
pub async fn run<T>(
    seconds: Option<u64>,
    cancel: &CancellationToken,
    engine: impl Future<Output = anyhow::Result<T>>,
) -> anyhow::Result<T> {
    let Some(seconds) = seconds else {
        return CURRENT.scope(None, engine).await;
    };
    run_with_activity(seconds, cancel, Activity::new(), engine).await
}

pub(crate) async fn run_with_activity<T>(
    seconds: u64,
    cancel: &CancellationToken,
    activity: Activity,
    engine: impl Future<Output = anyhow::Result<T>>,
) -> anyhow::Result<T> {
    anyhow::ensure!(seconds > 0, "turn_idle_timeout_secs must be positive");
    let mut updates = activity.0.changed.subscribe();
    let idle = Duration::from_secs(seconds);
    let engine = CURRENT.scope(Some(activity.clone()), engine);
    tokio::pin!(engine);
    loop {
        let deadline = activity.deadline(idle);
        tokio::select! {
            biased;
            result = &mut engine => return result,
            _ = updates.changed() => {},
            () = tokio::time::sleep_until(deadline) => {
                if Instant::now() < activity.deadline(idle) { continue; }
                // Let the existing cancellation path retain partial output,
                // account usage and settle local cleanup. Never replay effects.
                cancel.cancel();
                if let Ok(Err(error)) = tokio::time::timeout(Duration::from_secs(5), &mut engine).await
                    && zeroclaw_api::session_events::capture::is_capture_stopped(&error) {
                    return Err(error);
                }
                return Err(IdleTimeout { seconds }.into());
            }
        }
    }
}

#[cfg(test)]
mod tests;
