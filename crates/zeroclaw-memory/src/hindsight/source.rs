//! Host-owned admission and transcript visibility, independent of storage format.
use super::ledger::Ledger;
use zeroclaw_api::memory_traits::PrincipalScope;

/// Orient extraction as the resolved assistant's own memory, while keeping
/// the user's actions and preferences attributed to the user.
pub fn conversation_context(names: &zeroclaw_config::speakers::SpeakerNames) -> String {
    let assistant = &names.assistant;
    let user = &names.user;
    format!(
        "This is your conversation history. You are {assistant}; {user} is the user. \
         Extract memories from your own perspective, using \"I\" and \"my\" for your actions and experiences. \
         Keep {user}'s actions and preferences attributed to {user}."
    )
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversation_context_uses_resolved_names_and_separates_user_attribution() {
        let names = zeroclaw_config::speakers::SpeakerNames {
            assistant: "Nova".into(),
            user: "Alex".into(),
        };
        assert_eq!(
            conversation_context(&names),
            "This is your conversation history. You are Nova; Alex is the user. \
             Extract memories from your own perspective, using \"I\" and \"my\" for your actions and experiences. \
             Keep Alex's actions and preferences attributed to Alex."
        );
    }
}
