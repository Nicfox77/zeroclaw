//! Frozen branch selection derived only from committed branch origins.
use super::*;
use std::collections::{BTreeMap, HashMap, HashSet};

/// Resolve only immutable branch origins. SQLite indexes these lookups, so
/// the common linear conversation does not replay its archived content.
pub fn read_frozen_lineage(
    store: &dyn SessionEventStore,
    stream: &SessionId,
    branch: &BranchId,
    through: u64,
) -> Result<BTreeMap<BranchId, u64>, ReadError> {
    let mut origins = Vec::new();
    let mut seen = HashSet::new();
    let mut current = branch.clone();
    loop {
        if !seen.insert(current.clone()) {
            return Err(ReadError::Corrupt("cyclic branch lineage".into()));
        }
        let mut origin = None;
        for kind in [
            EventKind::BranchCreated,
            EventKind::SessionStarted,
            EventKind::DerivedSessionStarted,
        ] {
            if let Some(event) = store.read_latest_event(stream, &current, None, &kind)? {
                origin = Some(event);
                break;
            }
        }
        let origin = origin.ok_or_else(|| ReadError::Gap("branch origin missing".into()))?;
        let parent = match &origin.body.data {
            EventData::BranchCreated {
                parent_branch_id, ..
            } => Some(parent_branch_id.clone()),
            _ => None,
        };
        origins.push(origin);
        match parent {
            Some(parent) => current = parent,
            None => return frozen_lineage(&origins, stream, branch, through),
        }
    }
}

pub fn frozen_lineage<'a>(
    events: impl IntoIterator<Item = &'a SessionEvent>,
    stream: &SessionId,
    branch: &BranchId,
    through: u64,
) -> Result<BTreeMap<BranchId, u64>, ReadError> {
    let origins: HashMap<_, _> = events
        .into_iter()
        .filter(|e| {
            &e.session_id == stream
                && matches!(
                    e.body.kind(),
                    EventKind::SessionStarted
                        | EventKind::DerivedSessionStarted
                        | EventKind::BranchCreated
                )
        })
        .filter_map(|e| e.branch_id.as_ref().map(|b| (b.clone(), e)))
        .collect();
    let mut selected = BTreeMap::new();
    let mut seen = HashSet::new();
    let mut current = branch.clone();
    let mut bound = through;
    loop {
        if !seen.insert(current.clone()) {
            return Err(ReadError::Corrupt("cyclic branch lineage".into()));
        }
        let origin = origins
            .get(&current)
            .ok_or_else(|| ReadError::Gap("branch origin missing".into()))?;
        selected.insert(current.clone(), bound);
        match &origin.body.data {
            EventData::BranchCreated {
                parent_branch_id,
                parent_cursor,
            } => {
                if &parent_cursor.session_id != stream {
                    return Err(ReadError::Corrupt("cross-stream branch lineage".into()));
                }
                bound = bound.min(parent_cursor.sequence);
                current = parent_branch_id.clone();
            }
            _ => return Ok(selected),
        }
    }
}

pub fn includes(lineage: &BTreeMap<BranchId, u64>, event: &SessionEvent) -> bool {
    event
        .branch_id
        .as_ref()
        .and_then(|b| lineage.get(b))
        .is_some_and(|bound| event.sequence <= *bound)
}
