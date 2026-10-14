# Neural sampling without a waiting-time cap

Branch `codex/unbounded-neural-sampling` starts from the validated parametric
refactor (`b478a24`). It changes neural event generation, not likelihood formulas,
FullyNN's representation, MMD calculations or parametric simulators.

## Behavior

The next event can occur arbitrarily far beyond the dataset's largest observed
gap. `dtime_max` now initializes a search window; it is never used to clip a
returned delay. When a window is crossed without acceptance, the search continues
from its boundary and its width doubles. The intensity envelope is recomputed
on that new window. Proposals outside the current window are not accepted with
an envelope fitted to another interval. A window boundary is not an event.

Rejections continue until every requested draw is accepted. The old ten-round
fallback that returned an unaccepted proposal is removed. Completed draws retain
their values while pending draws continue. Random seeds reproduce the complete
calculation within the same supported environment.

```yaml
model_config:
  thinning_config:
    num_sample: 2
    num_exp: 10
    num_samples_boundary: 200
    over_sample_rate: 2.0
    max_sampling_rounds: 10000
```

The new positive integer `max_sampling_rounds` is a deterministic computation
safety budget, not a maximum delay or a wall-clock deadline. It can be increased.
The default permits 10000 proposal rounds; each round proposes `num_exp` times per
history position. The effective configuration and sampler protocol are recorded
in the run manifest.

## Safety and limits

`SamplingSafetyError` is raised on an exhausted budget, invalid/nonfinite
intensities, a detected envelope violation, nonfinite proposals or floating-point
stagnation. No rejected proposal is returned as an event. The exception preserves
the pending mask, accepted draws, offsets and search windows for diagnosis. The
pipeline records a failed phase; incomplete runs must not be interpreted as valid
statistical results. This is not automatic checkpoint/resume of mid-prediction
sampling, which remains unsupported.

Envelopes still come from sampled grids inflated by `over_sample_rate`. The
sampler checks them against evaluated proposals, but this does not mathematically
certify the envelope between grid points for arbitrary neural functions.
An observed violation is an error, not permission to accept with probability
greater than one. Model-specific analytical envelopes remain a separate accuracy
improvement. Neither a larger safety budget nor a denser grid proves correctness
for every possible intensity curve.

FullyNN retains its previously identified batch/grid scaling and bounded-hazard
issues. An integrable future intensity can give positive probability of no finite
next event. The new sampler exposes that limitation through safety failures;
the old fallback hid it by inventing a finite time. The validation fixture uses
a short explicit budget for this known failure and requires the phase to fail,
rather than reporting FullyNN as successfully simulated.

## Validation

```bash
uv run --frozen --no-sync python -m pytest tests/test_event_sampler.py tests/test_model_contracts.py
uv run --frozen --no-sync python -m scripts.unbounded_sampling_validation --output artifacts/unbounded-new
uv run --frozen --no-sync python -m scripts.independent_simulator_validation --size 4096 --output artifacts/laws-new
uv run --frozen --no-sync python -m scripts.conditional_reproducibility --model NHP --output artifacts/repeat-new
```

Use fresh output directories. Law tests include constant Poisson intensities with
long waits, increasing affine intensity checked via its analytical compensator,
more than ten rejected rounds and zero-intensity safety failures. Independent
model references additionally exercise a small initial window and many rejections.
The model audit tests all twelve catalogue entries, compares Hawkes and
SelfCorrecting to the last parametric reference and checks unchanged neural
weights/likelihood/intensity calculations while recording intended changes to
predictions, simulations and statistics. FullyNN's explicit failure is reported.

The optional workflow `unbounded-sampling.yml` runs these checks on Linux when
dispatched or requested by `[unbounded-sampling]` on a push to this branch. It
also verifies exact NHP pipeline repetition, checkpoint replay and training
resume. CUDA tests skip explicitly without GPU hardware. Historical reference
files are not overwritten or replaced.
