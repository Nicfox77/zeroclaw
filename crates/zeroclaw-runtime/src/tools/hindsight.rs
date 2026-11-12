//! Reflection uses the same principal-bound memory handle as automatic recall.
use async_trait::async_trait;
use serde_json::{Value, json};
use std::sync::Arc;
use zeroclaw_api::tool::{Tool, ToolResult};
use zeroclaw_config::policy::SecurityPolicy;
use zeroclaw_memory::Memory;
#[derive(Clone, Copy)]
pub enum Operation {
    Reflect,
}
pub struct HindsightTool {
    memory: Arc<dyn Memory>,
    description: String,
}
impl HindsightTool {
    pub fn new(
        memory: Arc<dyn Memory>,
        _security: Arc<SecurityPolicy>,
        _operation: Operation,
    ) -> Self {
        Self {
            memory,
            description: crate::i18n::get_required_cli_string("cli-hindsight-tool-reflect"),
        }
    }
}
zeroclaw_api::tool_attribution!(HindsightTool, zeroclaw_api::attribution::ToolKind::Memory);
#[async_trait]
impl Tool for HindsightTool {
    fn name(&self) -> &str {
        "hindsight_reflect"
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn parameters_schema(&self) -> Value {
        json!({"type":"object","properties":{"query":{"type":"string"}},"required":["query"],"additionalProperties":false})
    }
    async fn execute(&self, args: Value) -> anyhow::Result<ToolResult> {
        anyhow::ensure!(
            matches!(
                self.memory.role(),
                zeroclaw_api::attribution::Role::Memory(
                    zeroclaw_api::attribution::MemoryKind::Hindsight
                )
            ),
            "hindsight_backend_required"
        );
        let query = args
            .get("query")
            .and_then(Value::as_str)
            .filter(|q| !q.trim().is_empty())
            .ok_or_else(|| {
                anyhow::Error::msg(crate::i18n::get_required_cli_string(
                    "cli-hindsight-query-required",
                ))
            })?;
        let entries = self.memory.reflect(query).await?;
        Ok(ToolResult {
            success: true,
            output: serde_json::to_string(&entries)?.into(),
            error: None,
        })
    }
}
