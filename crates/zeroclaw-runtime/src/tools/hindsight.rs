//! Native memory tools using the same scoped, screened memory binding as recall.
use async_trait::async_trait;
use serde_json::{Value, json};
use std::sync::Arc;
use zeroclaw_api::tool::{Tool, ToolResult};
use zeroclaw_config::policy::{SecurityPolicy, ToolOperation};
use zeroclaw_memory::{Memory, MemoryCategory};
#[derive(Clone, Copy)]
pub enum Operation {
    Recall,
    Reflect,
    Retain,
}
pub struct HindsightTool {
    memory: Arc<dyn Memory>,
    security: Arc<SecurityPolicy>,
    operation: Operation,
    description: String,
}
impl HindsightTool {
    pub fn new(
        memory: Arc<dyn Memory>,
        security: Arc<SecurityPolicy>,
        operation: Operation,
    ) -> Self {
        let key = match operation {
            Operation::Recall => "cli-hindsight-tool-recall",
            Operation::Reflect => "cli-hindsight-tool-reflect",
            Operation::Retain => "cli-hindsight-tool-retain",
        };
        Self {
            memory,
            security,
            operation,
            description: crate::i18n::get_required_cli_string(key),
        }
    }
}
zeroclaw_api::tool_attribution!(HindsightTool, zeroclaw_api::attribution::ToolKind::Memory);
#[async_trait]
impl Tool for HindsightTool {
    fn name(&self) -> &str {
        match self.operation {
            Operation::Recall => "hindsight_recall",
            Operation::Reflect => "hindsight_reflect",
            Operation::Retain => "hindsight_retain",
        }
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn parameters_schema(&self) -> Value {
        match self.operation {
            Operation::Retain => {
                json!({"type":"object","properties":{"key":{"type":"string"},"content":{"type":"string"}},"required":["key","content"],"additionalProperties":false})
            }
            _ => {
                json!({"type":"object","properties":{"query":{"type":"string"}},"required":["query"],"additionalProperties":false})
            }
        }
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
        let result = match self.operation {
            Operation::Retain => {
                self.security
                    .enforce_tool_operation(ToolOperation::Act, self.name())
                    .map_err(anyhow::Error::msg)?;
                let key = args.get("key").and_then(Value::as_str).ok_or_else(|| {
                    anyhow::Error::msg(crate::i18n::get_required_cli_string(
                        "cli-hindsight-key-required",
                    ))
                })?;
                let content = args.get("content").and_then(Value::as_str).ok_or_else(|| {
                    anyhow::Error::msg(crate::i18n::get_required_cli_string(
                        "cli-hindsight-content-required",
                    ))
                })?;
                self.memory
                    .store(key, content, MemoryCategory::Core, None)
                    .await?;
                crate::i18n::get_required_cli_string("cli-hindsight-retained")
            }
            _ => {
                let query = args
                    .get("query")
                    .and_then(Value::as_str)
                    .filter(|q| !q.trim().is_empty())
                    .ok_or_else(|| {
                        anyhow::Error::msg(crate::i18n::get_required_cli_string(
                            "cli-hindsight-query-required",
                        ))
                    })?;
                let entries = match self.operation {
                    Operation::Reflect => self.memory.reflect(query).await?,
                    _ => self.memory.recall(query, 8, None, None, None).await?,
                };
                serde_json::to_string(&entries)?
            }
        };
        Ok(ToolResult {
            success: true,
            output: result.into(),
            error: None,
        })
    }
}
