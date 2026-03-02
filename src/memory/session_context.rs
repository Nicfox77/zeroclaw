use std::future::Future;

use tokio::task_local;

task_local! {
    static MEMORY_SESSION_ID: Option<String>;
}

/// Run a future with a scoped memory session id available to tools.
pub async fn scope_session_id<F>(session_id: Option<&str>, fut: F) -> F::Output
where
    F: Future,
{
    MEMORY_SESSION_ID
        .scope(session_id.map(str::to_string), fut)
        .await
}

/// Returns the currently scoped memory session id, if any.
pub fn current_session_id() -> Option<String> {
    MEMORY_SESSION_ID.try_with(Clone::clone).ok().flatten()
}
