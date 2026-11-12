//! Host-owned admission and transcript visibility, independent of storage format.
use super::ledger::Ledger;
use zeroclaw_api::memory_traits::PrincipalScope;

/// This authority is supplied by trusted composition, never by tool arguments.
/// Scoped Memory methods receive their principal through the existing binding;
/// hosts with a canonical execution scope may also support unscoped methods.
pub trait TranscriptAuthority: Send + Sync {
    fn admitted_scope(&self) -> anyhow::Result<PrincipalScope> {
        anyhow::bail!("hindsight_principal_scope_required")
    }
    /// Reject reads while source erasure or authorization reconciliation is due.
    fn check_sources(&self, ledger: &Ledger, bank: &str) -> anyhow::Result<()>;
}

/// Explicit notes can operate without a transcript source. An existing source
/// must never become readable merely because its adapter was not installed.
pub struct UnboundTranscriptAuthority;
impl TranscriptAuthority for UnboundTranscriptAuthority {
    fn check_sources(&self, ledger: &Ledger, bank: &str) -> anyhow::Result<()> {
        anyhow::ensure!(
            !ledger.has_transcript_sources(bank)?,
            "hindsight_transcript_authority_required"
        );
        Ok(())
    }
}
