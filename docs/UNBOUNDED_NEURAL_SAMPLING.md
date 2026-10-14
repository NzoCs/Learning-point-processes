# Neural sampling without a waiting-time cap

Branch `codex/unbounded-neural-sampling` starts from the validated parametric
refactor (`b478a24`). It changes neural event generation and corrects FullyNN's
pointwise differentiation and compensator. The network architecture, MMD
calculations, parametric simulators and historical baseline files are preserved.

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

## FullyNN correction and absence of a next event

The paper defines a pointwise derivative (equation 13, section 2.4,
https://arxiv.org/pdf/1905.09690). PyTorch must differentiate the sum of the
independent outputs, not their mean: the previous mean divided each intensity
by batch * sequence * grid size. Last-step queries now differentiate only the
last history. The loss uses `Z(t)-Z(0)`, the integral of that derivative from
zero; this preserves the corrected intensity and makes the compensator zero
at zero. Retrain FullyNN; old fitted results are intentionally changed.

For total intensity lambda, survival is `exp(-H(t))`. If its integral to infinity
is finite, no next event exists with probability `exp(-H(infinity))`. For
`lambda(t)=2 exp(-t)`, this probability is `exp(-2)`. RMTPP's negative temporal
slopes and FullyNN's bounded tanh network are preserved.

Both models now expose an integrated intensity and its analytic infinite-time
limit. Independent Exp(1) thresholds above the limit produce infinite delays;
finite thresholds are inverted by bracketing and bisection without a time cap.
RMTPP integrates its existing intensity cap, including zero and mixed slopes.
No failed computation or numerical tail cutoff is interpreted as absence.

Simulation terminates only the affected row and continues other rows. Absent
events remain masked padding; infinite times are never exported as events.
`Batch.no_next_event_mask` and `simulations/sequence_status.jsonl` record actual
termination, including rows with zero future events. Entirely empty event
outputs remain valid empty Parquet tables.

One-step forecasts explicitly condition on existence of the next event and
include `no_event_probability` separately. Validation/test logs report its mean
over valid histories. The finite conditional forecast is not an unconditional
mean, which is infinite when absence has positive probability. Simulations
retain the unconditional law. Scientific calibration on terminating processes
still requires a separate validation.

## Validation

```bash
uv run --frozen --no-sync python -m pytest tests/test_hazard_sampling.py tests/test_event_sampler.py tests/test_model_contracts.py
uv run --frozen --no-sync python -m scripts.unbounded_sampling_validation --output artifacts/unbounded-new
uv run --frozen --no-sync python -m scripts.independent_simulator_validation --size 4096 --output artifacts/laws-new
uv run --frozen --no-sync python -m scripts.conditional_reproducibility --model NHP --output artifacts/repeat-new
```

Use fresh output directories. Law tests include constant Poisson intensities with
long waits, increasing affine intensity checked via its analytical compensator,
more than ten rejected rounds and zero-intensity safety failures. Independent
model references additionally exercise a small initial window and many rejections.
The model audit tests all twelve catalogue entries, compares Hawkes and
SelfCorrecting to the last parametric reference and checks neural likelihood,
intensity and gradients with the reviewed historical weights fixed. It separately
records changes in newly fitted weights, predictions, simulations and statistics.
Refitted weights need not match historical CPU profiles exactly, particularly
near numerically cancelling attention-bias gradients. That comparison is retained
as evidence rather than silently replacing profiles or relaxing tolerances.
FullyNN's intended formula changes are audited separately. Independent checks
test finite differences, batch/grid invariance, the atom at infinity and finite
CDFs, RMTPP compensators against quadrature including capped intensities, and
row-wise termination with preservation of empty sequence records. The eleven
historical models plus Poisson must complete train/test/predict successfully.

The optional workflow `unbounded-sampling.yml` runs these checks on Linux when
dispatched or requested by `[unbounded-sampling]` on a push to this branch. It
also verifies exact ANHN and NHP pipeline repetition, checkpoint replay and training
resume. CUDA tests skip explicitly without GPU hardware. Historical reference
files are not overwritten or replaced.
