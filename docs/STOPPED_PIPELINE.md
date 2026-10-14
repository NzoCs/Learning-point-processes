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

## Validated Linux results — 14 October 2026

Code `8295bd55b15296a5d6b92b01e6fe4d3e20e2e468` passes
[the Linux workflow](https://github.com/NzoCs/Learning-point-processes/actions/runs/38589688373)
and [lint](https://github.com/NzoCs/Learning-point-processes/actions/runs/38589688065).
The unit suite reports **305 passed, 7 skipped, 2 deselected**, with **84.26%**
coverage. All eleven historical models plus Poisson pass ordinary and conditional
integration; FullyNN/RMTPP also pass the known-terminal fixtures, exact same-seed
repetition, checkpoint prediction replay and epoch-boundary training resume.

Against reviewed parent `acd6f374cedfad47081b65d7cbe082ba1dc007cb` rerun on the
same Linux CPU and locked environment, all numerical fields of the other nine
historical models and Poisson agree at the existing `rtol=1e-5`, `atol=1e-6`.
FullyNN/RMTPP trained weights, losses, gradients, predictions, simulated events
and termination statuses also agree. Their observation statistics change.

The conditional comparison below uses the same observed/generated futures and
the same starting permutation RNG state, with 99 permutations. Training data are
separate from evaluated futures. The known-terminal fixture includes observed
future lengths 0, 1, 0 and 2 with explicit terminal status, followed by four
three-event futures.

| Model / fixture | Historical truncation p-value | Complete protocol p-value |
| --- | ---: | ---: |
| FullyNN / ordinary held-out data | 1.00 | 0.05 |
| RMTPP / ordinary held-out data | 0.52 | 0.38 |
| FullyNN / known-terminal observed futures | 1.00 | 0.10 |
| RMTPP / known-terminal observed futures | 0.42 | 0.05 |

These are small integration fixtures, not estimates of scientific accuracy or
false-positive rates. FullyNN generates one future event across eight histories;
RMTPP generates 24. The complete protocol retains this difference instead of
erasing finite events at a zero minimum horizon.

In the ordinary history/continuation diagnostic, mean MMD changes from 3.317620
to 16.372105 for FullyNN and from 10.746567 to 15.196381 for RMTPP. These values
use different observation contracts and do not establish improved model fit or
conditional calibration of that diagnostic.

[Raw audit, before/after snapshots, replay results and provenance](validation/stopped-pipeline-20261014/provenance.json)
are archived without replacing historical reference files. The downloaded Linux
artifact SHA256 is `7deb5ae710d0af9f6b5b83d92680f1841a1f37d502f84e96d585c2060cbf216f`.
This validation uses CPU Linux; Ruche/CUDA remains untested.
