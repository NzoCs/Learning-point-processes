# Complete stopped observations in production pipelines

FullyNN and RMTPP automatically select `complete_stopped_event_budget_v1` in
their prediction statistics. The other models retain their existing observation
protocol. The model architectures, loss functions, sampling laws, prediction
methods, training seeds and historical reference files are unchanged by this
wiring. The new terminal metadata is used for observation/statistics and does
not add a terminal-survival term to the training objective.

## Conditional held-out futures

```yaml
simulation_config:
  seed: 42
  conditional_evaluation:
    history_events: 2
    future_events: 3
```

The model receives only the two historical events. The observed three-event
future and its terminal flag stay outside the model input. Both groups retain
their complete futures, including empty ones, and use the paired joint kernel
with the terminal-flag Gaussian. Within-history swaps, product-group aggregation,
disjoint sequence ID checks and the existing plus-one p-value remain in place.

A source with at least five valid events reaches this three-event observation
quota. Its terminal flag for this observation is false; any later source end is
outside the observation budget. A source with only two, three or four valid
events needs an explicitly known end:

```json
{
  "sequence_id": "known-ended-sequence",
  "dim_process": 2,
  "time_since_start": [0.1, 0.3],
  "time_since_last_event": [0.1, 0.2],
  "type_event": [0, 1],
  "no_next_event": true
}
```

This row has an empty held-out future and is kept in the statistics and sequence
identity checks. JSON/Hugging Face inputs and the dataset/collator preserve the
boolean metadata. Missing/null metadata means no known terminal status. A short
unknown future fails before simulation, rather than being filtered or labeled
terminal. Truncating a source carrying a known terminal flag also fails: that
operation would lose the actual end. Finite stored times and masked padding are
retained; absent events are never exported as infinite event times.

## Ordinary history/continuation diagnostic

Without `conditional_evaluation`, the existing history-versus-continuation
comparison remains a diagnostic, not a calibrated conditional goodness-of-fit
test. FullyNN/RMTPP now retain complete paths and terminal flags in this comparison
as well. An ordinary complete MMD uses pooled normalization and full label
permutations; it does not silently become a matched-history test. The manifest
continues to mark its conditional calibration as false. Short source rows need
known terminal status to distinguish them from unknown observation censoring.

The run manifest and statistical metadata record the effective complete
observation protocol. Calibration of the paired design is documented in
[the scientific report](STOPPED_CALIBRATION.md). Real data collected over a finite
time window require a matched time-window observation design; quiet time alone
does not establish permanent termination.

## Audit

```bash
uv run --frozen --no-sync python -m scripts.stopped_pipeline_validation \
  --output artifacts/stopped-pipeline-new
```

The optional Linux workflow runs train/test/predict for all eleven historical
models plus Poisson, and their conditional pipelines on separately generated
train/dev/test data. It first reruns the reviewed pre-wiring commit `acd6f37`
on the same runner with the same locked runtime, independently loaded from its
own checkout. This separates native-CPU refitting drift from the wiring change;
old archived snapshots are not replaced and tolerances are not widened.
Ordinary output snapshots for the nine unaffected historical models and Poisson
must match every numerical field of that same-runner reference. For FullyNN/RMTPP,
fitted weights, losses, gradients, predictions,
trajectories and sequence termination statuses must match their pre-wiring
snapshots; only the observation protocol and prediction statistics change.

The conditional collector also evaluates a historical-truncation control on the
same observed/simulated data with exactly the same starting null RNG state. It
records all before/after null draws and p-values without changing the model or
simulation streams. Additional CLI fixtures include explicitly terminal empty
and partial observed futures. FullyNN/RMTPP conditional repetition, checkpoint
replay and epoch-boundary resume are checked separately. No Ruche is required.
