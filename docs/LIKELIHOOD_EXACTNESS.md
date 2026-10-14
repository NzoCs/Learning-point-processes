# Likelihood, finite-window calibration and interruption contracts

The generic neural sampler's sampled-grid envelope is preserved at the user's
request. This change makes no new claim about its mathematical certification.

## Event counts and analytic RMTPP integration

FullyNN and the shared neural likelihood utility count valid scored events from
their masks, independently of the log-intensity value. The shared utility also
masks event contributions. Per-event training rejects a batch with no scored
events instead of dividing by zero.

RMTPP keeps its marked intensity `exp(min(a + w*t, log(1e5)))`. An exact piecewise
integral replaces Monte Carlo quadrature in training/validation/test losses and
is shared with the simulation hazard. A local `expm1(z)/z` series preserves the
slope derivative at zero: `dH/dw = exp(a)*t**2/2` below the cap. Negative, zero and
positive slopes, long durations and capped plateaus are tested against SciPy
quadrature for both values and gradients. No slope sign constraint is imposed.

RMTPP training results can change because approximate gradients are replaced and
integration sampling no longer consumes random draws. The architecture/intensity
are unchanged. The optional Linux audit reruns parent `27ac222` on the same CPU,
compares all numerical fields of other models at existing tolerances and archives
RMTPP's intentional differences without replacing historical references.

## Explicit permanent ends in learning

The transition likelihood still conditions on the first recorded event. For
FullyNN/RMTPP only, source `no_next_event: true` adds the negative log probability
of no next event: `H(infinity | last valid history)`. Padded states cannot supply
that history. The event counter still counts transitions; the end term changes
the likelihood numerator. Training requires at least one scored transition per
batch for per-event normalization.

Unknown/false status adds no end term. Event quotas, file ends and quiet finite
windows must not be declared permanent ends. If a known end has zero probability
under RMTPP's current slopes, its exact negative log-likelihood is infinite.
Training rejects that nonfinite objective explicitly, without forcing parameters
to change. Validation/test may report the infinite score of an incompatible
trajectory; conditional permutation testing can still compare its future.

## Predeclared finite windows

`ConditionalMMDTwoSampleTest(..., observation_horizon=T)` retains every event in
the same positive finite window for both futures, including empty ones. It uses
within-history swaps and independent-batch product-group aggregation, without a
permanent-end kernel. It rejects terminal flags and events beyond T. Window and
stopped-event-budget designs are mutually exclusive.

This explicit statistical API does not automatically convert event-count source
files: callers must establish complete observation through T. The existing event-
count pipeline remains the default. `scripts.fixed_window_calibration` observes
production simulations until T and compares them with an independent NumPy/SciPy
decoder/inversion/RNG, sharing the frozen Torch history encoder. Computation
budget exhaustion fails the experiment and never becomes observation censoring.

`scripts.likelihood_calibration_validation` fits each model for one and two epochs
on separate train/dev data and freezes two distinct last checkpoints. Separate
histories/futures feed both observation designs. Per experiment: 300 H0 trials,
100 per H1 alternative, 99 permutations and batches of 16/11/5 histories. The
familywise H0 excess decision uses a predeclared one-sided binomial threshold
`0.01/16`. Raw p-values, intervals and power are retained. This does not include
parameter-estimation uncertainty or certify arbitrary fitted checkpoints.

## Interruptions and exact restart

New training checkpoints record whether the epoch is completed. Mid-epoch
training resume is explicitly rejected because the shuffled loader cursor is
not checkpointed. Completed-epoch resume remains supported, including legacy
completed-epoch checkpoints from the previous validated runs.

Prediction failure marks its manifest phase as failed. The reproducibility audit
injects an interruption after one exported batch, then restarts the whole
prediction in a fresh output directory from the same checkpoint/seed. Events,
statistics, all null draws and p-values must match exactly. This is a full
restart, not continuation of a partial export.

The optional Linux workflow includes all eleven historical models plus Poisson,
ordinary/conditional/known-terminal integrations, repeats/replay/resume and the
two-checkpoint calibrations. Final measured evidence is archived after validation.
Ruche/CUDA/DDP is outside this CPU validation.
