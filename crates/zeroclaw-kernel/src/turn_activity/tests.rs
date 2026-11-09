use super::*;

#[tokio::test(start_paused = true)]
async fn waiting_turn_observes_background_progress_without_creating_heartbeats() {
    for progressing in [true, false] {
        let job = Activity::new();
        let worker = job.clone();
        let child = tokio::spawn(async move {
            let cancel = CancellationToken::new();
            run_with_activity(1, &cancel, worker, async {
                for _ in 0..8 {
                    tokio::time::sleep(Duration::from_millis(500)).await;
                    if progressing {
                        Activity::progress_current();
                    }
                }
                Ok(())
            })
            .await
        });
        let cancel = CancellationToken::new();
        let result = run(Some(1), &cancel, job.observe(async { child.await? })).await;
        if progressing {
            result.unwrap();
            assert!(!cancel.is_cancelled());
        } else {
            assert!(result.unwrap_err().is::<IdleTimeout>());
            assert!(cancel.is_cancelled());
        }
    }
}

#[tokio::test(start_paused = true)]
async fn continuing_progress_can_outlive_many_idle_windows() {
    let cancel = CancellationToken::new();
    let started = Instant::now();
    let result = run(Some(3), &cancel, async {
        for _ in 0..12 {
            tokio::time::sleep(Duration::from_secs(2)).await;
            Activity::progress_current();
        }
        Ok("finished")
    })
    .await
    .unwrap();
    assert_eq!(result, "finished");
    assert_eq!(started.elapsed(), Duration::from_secs(24));
    assert!(!cancel.is_cancelled());
}

#[tokio::test(start_paused = true)]
async fn stalled_work_is_cancelled_after_the_last_real_progress() {
    let cancel = CancellationToken::new();
    let started = Instant::now();
    let error = run(Some(3), &cancel, async {
        for _ in 0..3 {
            tokio::time::sleep(Duration::from_secs(2)).await;
            Activity::progress_current();
        }
        cancel.cancelled().await;
        Err::<(), _>(anyhow::Error::msg("cooperative stop"))
    })
    .await
    .unwrap_err();
    assert_eq!(error.downcast_ref::<IdleTimeout>().unwrap().seconds, 3);
    assert_eq!(started.elapsed(), Duration::from_secs(9));
    assert!(cancel.is_cancelled());
}

#[tokio::test(start_paused = true)]
async fn bounded_quiet_operation_retains_its_own_budget() {
    let cancel = CancellationToken::new();
    run(Some(2), &cancel, async {
        let wait = Activity::current()
            .unwrap()
            .bounded_wait(Duration::from_secs(12));
        tokio::time::sleep(Duration::from_secs(10)).await;
        drop(wait);
        tokio::time::sleep(Duration::from_secs(1)).await;
        Ok(())
    })
    .await
    .unwrap();
}

#[tokio::test(start_paused = true)]
async fn a_bounded_wait_is_not_an_infinite_heartbeat() {
    let cancel = CancellationToken::new();
    let started = Instant::now();
    let error = run(Some(2), &cancel, async {
        let _wait = Activity::current()
            .unwrap()
            .bounded_wait(Duration::from_secs(3));
        cancel.cancelled().await;
        Err::<(), _>(anyhow::Error::msg("cooperative stop"))
    })
    .await
    .unwrap_err();
    assert!(error.is::<IdleTimeout>());
    // The existing operation budget plus bounded cleanup grace, not forever.
    assert_eq!(started.elapsed(), Duration::from_secs(8));
}

#[tokio::test(start_paused = true)]
async fn completed_operation_releases_unused_quiet_allowance() {
    let cancel = CancellationToken::new();
    let started = Instant::now();
    let error = run(Some(2), &cancel, async {
        let wait = Activity::current()
            .unwrap()
            .bounded_wait(Duration::from_secs(100));
        tokio::time::sleep(Duration::from_secs(1)).await;
        drop(wait);
        cancel.cancelled().await;
        Err::<(), _>(anyhow::Error::msg("cooperative stop"))
    })
    .await
    .unwrap_err();
    assert!(error.is::<IdleTimeout>());
    assert_eq!(started.elapsed(), Duration::from_secs(3));
}

#[tokio::test(start_paused = true)]
async fn parallel_waits_retain_the_later_live_deadline() {
    let cancel = CancellationToken::new();
    run(Some(2), &cancel, async {
        let activity = Activity::current().unwrap();
        let short = activity.bounded_wait(Duration::from_secs(1));
        let long = activity.bounded_wait(Duration::from_secs(10));
        tokio::join!(
            async {
                tokio::time::sleep(Duration::from_secs(1)).await;
                drop(short);
            },
            async {
                tokio::time::sleep(Duration::from_secs(8)).await;
                drop(long);
            }
        );
        Ok(())
    })
    .await
    .unwrap();
}

#[tokio::test(start_paused = true)]
async fn no_opt_in_preserves_unsupervised_legacy_engine_and_masks_parent() {
    let cancel = CancellationToken::new();
    run(Some(2), &cancel, async {
        run(None, &cancel, async {
            assert!(Activity::current().is_none());
            Ok(())
        })
        .await
    })
    .await
    .unwrap();
}

#[tokio::test(start_paused = true)]
async fn capture_failure_during_idle_cleanup_stays_fatal() {
    let cancel = CancellationToken::new();
    let error = run(Some(2), &cancel, async {
        cancel.cancelled().await;
        Err::<(), _>(zeroclaw_api::session_events::capture::capture_stopped(
            anyhow::Error::msg("fixture"),
        ))
    })
    .await
    .unwrap_err();
    assert!(zeroclaw_api::session_events::capture::is_capture_stopped(
        &error
    ));
    assert!(!error.is::<IdleTimeout>());
}
