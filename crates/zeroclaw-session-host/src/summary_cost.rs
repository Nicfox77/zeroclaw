//! Summary inference uses the existing ledger and live operator rate sheets.
use super::*;
use zeroclaw_config::cost::{BudgetCheck, CostTracker, TokenUsage};
use zeroclaw_providers::pricing::ModelRates;

fn rates(config: &zeroclaw_config::schema::Config, provider: &str, model: &str) -> ModelRates {
    let (family, alias) = provider.split_once('.').unwrap_or((provider, ""));
    let mut values = config
        .providers
        .models
        .find(family, alias)
        .map(|entry| entry.pricing.clone())
        .unwrap_or_default();
    for (kind, name, rate) in config.cost.rates.providers.models.iter_entries() {
        if kind != family {
            continue;
        }
        for (suffix, value) in [
            ("input", rate.input_per_mtok),
            ("output", rate.output_per_mtok),
            ("cached_input", rate.cached_input_per_mtok),
            ("cache_write", rate.cache_write_per_mtok),
        ] {
            if let Some(value) = value {
                values.insert(format!("{name}.{suffix}"), value);
            }
        }
    }
    let configured = zeroclaw_providers::pricing::model_id_candidates(model).fold(
        ModelRates::default(),
        |result, name| {
            let flat = values.get(name).copied();
            result.or(ModelRates {
                input_per_mtok: values.get(&format!("{name}.input")).copied().or(flat),
                output_per_mtok: values.get(&format!("{name}.output")).copied().or(flat),
                cached_input_per_mtok: values.get(&format!("{name}.cached_input")).copied(),
                cache_write_per_mtok: values.get(&format!("{name}.cache_write")).copied(),
            })
        },
    );
    configured.or(zeroclaw_providers::pricing::lookup(
        &zeroclaw_providers::pricing::current_snapshot(),
        provider,
        model,
    )
    .copied()
    .unwrap_or_default())
}

pub(super) fn admit(
    live: &LiveConfigHandle,
    agent: &str,
    provider: &str,
    model: &str,
    messages: &[zeroclaw_api::model_provider::ChatMessage],
) -> anyhow::Result<Option<CostTracker>> {
    let config = live.read();
    let Some(tracker) = CostTracker::get_or_init_global(
        config.cost.clone(),
        &config.data_dir,
        &config.channels.session_backend,
    ) else {
        return Ok(None);
    };
    let profile = config
        .runtime_profile_for_agent(agent)
        .ok_or_else(|| anyhow::Error::msg("summary_deck_profile_unavailable"))?;
    let ceiling = if profile.max_cost_per_day_cents == 0 {
        config.cost.daily_limit_usd
    } else {
        profile.max_cost_per_day_cents as f64 / 100.0
    };
    let tracker = tracker.derived_for_agent(agent, ceiling);
    let price = rates(&config, provider, model);
    let estimate = messages
        .iter()
        .map(|message| message.content.chars().count().div_ceil(4) + 4)
        .sum::<usize>() as f64
        * price.input_per_mtok.unwrap_or(0.0)
        / 1_000_000.0
        + profile.summary_deck.max_summary_tokens as f64 * price.output_per_mtok.unwrap_or(0.0)
            / 1_000_000.0;
    anyhow::ensure!(
        !matches!(
            tracker.check_budget(estimate)?,
            BudgetCheck::Exceeded { .. }
        ),
        "summary_deck_cost_budget_exceeded"
    );
    Ok(Some(tracker))
}

pub(super) fn record(
    live: &LiveConfigHandle,
    tracker: Option<&CostTracker>,
    agent: &str,
    session: &str,
    report: zeroclaw_providers::dispatch::AccountedCallReport,
) -> anyhow::Result<()> {
    let Some(tracker) = tracker else {
        return Ok(());
    };
    let config = live.read();
    for attempt in report.attempts() {
        use zeroclaw_providers::dispatch::AttemptUsageOutcome;
        let usage = match attempt.outcome() {
            AttemptUsageOutcome::Complete(usage)
            | AttemptUsageOutcome::OutcomeUnknown {
                observed: Some(usage),
            } => usage,
            _ => continue,
        };
        let input = usage.input_tokens.unwrap_or(0);
        let output = usage.output_tokens.unwrap_or(0);
        let cached = usage.cached_input_tokens.unwrap_or(0).min(input);
        let write = usage
            .cache_creation_input_tokens
            .unwrap_or(0)
            .min(input.saturating_sub(cached));
        let price = rates(&config, attempt.provider_ref(), attempt.model());
        let mut cost = TokenUsage::new_with_cache_write(
            format!("{}/{}", attempt.provider_ref(), attempt.model()),
            input,
            cached,
            write,
            output,
            price.input_per_mtok.unwrap_or(0.0),
            price.cached_input_per_mtok.unwrap_or(0.0),
            price.cache_write_per_mtok.unwrap_or(0.0),
            price.output_per_mtok.unwrap_or(0.0),
        );
        cost.unpriced_tokens = if price.input_per_mtok.is_none() {
            input.saturating_sub(cached).saturating_sub(write)
        } else {
            0
        } + if price
            .cached_input_per_mtok
            .or(price.input_per_mtok)
            .is_none()
        {
            cached
        } else {
            0
        } + if price
            .cache_write_per_mtok
            .or(price.input_per_mtok)
            .is_none()
        {
            write
        } else {
            0
        } + if price.output_per_mtok.is_none() {
            output
        } else {
            0
        };
        cost.pricing_available = cost.unpriced_tokens == 0;
        tracker.record_usage_attributed(cost, Some(agent), None, Some(session.into()))?;
    }
    Ok(())
}
