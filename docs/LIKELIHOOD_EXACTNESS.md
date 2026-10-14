# Likelihood, finite-window calibration and interruption contracts

The generic neural sampler's sampled-grid envelope is preserved at the user's
request. This change makes no new claim about its mathematical certification.

## Event counts and analytic RMTPP integration

FullyNN and the shared neural likelihood utility count valid scored events from
their masks, independently of the log-intensity value. The shared utility also
masks event contributions. Per-event training rejects a batch with no scored
events instead of dividing by zero. Training/validation/test steps move valid
events before masked padding, preserving actual event order. Direct FullyNN and
RMTPP losses do the same. Left/right padding must produce the same likelihood,
event counts and gradients; the first valid event is conditioning history.

RMTPP keeps its marked intensity `exp(min(a + w*t, log(1e5)))`. An exact piecewise
integral replaces Monte Carlo quadrature in training/validation/test losses and
is shared with the simulation hazard. A local `expm1(z)/z` series preserves the
slope derivative at zero: `dH/dw = exp(a)*t**2/2` below the cap. Negative, zero and
positive slopes, long durations and capped plateaus are tested against SciPy
quadrature for both values and gradients. No slope sign constraint is imposed.
Endpoint-rate differences avoid `0 * infinity` when a very small initial
intensity grows over a long interval. The reproduced case `a=-1000, w=1, t=2000`
must have a finite integral and finite gradients, despite `exp(a)` underflow.

RMTPP training results can change because approximate gradients are replaced and
integration sampling no longer consumes random draws. The architecture/intensity
are unchanged. The optional Linux audit reruns parent `27ac222` on the same CPU,
compares all numerical fields of other models at existing tolerances and archives
RMTPP's intentional differences without replacing historical references.
The removed loss-sampling draws also shift the Monte Carlo one-step forecast
stream during validation/test. Small forecast-metric differences therefore do
not measure only the change in trained parameters.

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

## Measured results — 14 October 2026

Code `5f37648d46ba19ab7161b982d163929846f74b3e` passes
[the full Linux workflow](https://github.com/NzoCs/Learning-point-processes/actions/runs/38604105572)
and [lint](https://github.com/NzoCs/Learning-point-processes/actions/runs/38604105502):
**327 passed, 7 skipped, 2 deselected; 84.35% coverage**. All eleven historical
models plus Poisson pass ordinary/conditional integration. The other eleven
catalogue entries, including FullyNN on nonterminal reference data, preserve
all audited numerical fields at `rtol=1e-5`, `atol=1e-6`. RMTPP's intentional
changes are retained in before/after snapshots. Both stopped models pass exact
repeat, checkpoint replay, completed-epoch resume and complete prediction restart
after an injected interruption; the partial prediction remains explicitly failed.

Small ordinary integration fixture, same-runner before/after:

| RMTPP metric | Monte Carlo loss parent | Analytic loss |
| --- | ---: | ---: |
| Test loss | 1.248142 | 1.246912 |
| Time MAE | 0.377479 | 0.366567 |
| Time RMSE | 0.506855 | 0.507372 |
| Mark accuracy | 50% | 50% |

These eight-sequence results are regression evidence, not a scientific model
ranking. Known terminal sources and previously misinterpreted left padding can
also intentionally change losses; the nonterminal equal-length fixture cannot
establish unchanged results for those newly corrected cases.

The four distinct fitted parameter states each pass two independent-oracle
experiments: **300 H0 trials and 100 trials per H1 alternative**, two marks,
99 permutations, batches 16/11/5, event budget 6 or time horizon 2. This totals
2400 H0 trials and 1600 H1 trials across eight experiments. Aggregated H0 rates:

| Model / training epochs | Stopped quota | Fixed window |
| --- | ---: | ---: |
| FullyNN / 1 | 5.67% (95% CI 3.57–8.89%) | 4.33% (2.55–7.27%) |
| FullyNN / 2 | 3.33% (1.82–6.03%) | 2.33% (1.13–4.74%) |
| RMTPP / 1 | 4.33% (2.55–7.27%) | 3.67% (2.06–6.45%) |
| RMTPP / 2 | 4.33% (2.55–7.27%) | 5.67% (3.57–8.89%) |

No H0 excess is detected under the predeclared family threshold `0.01/16`.
The largest single-batch rate is RMTPP/2 window at **7.0%**, CI **4.62–10.46%**;
it remains in the raw evidence. These finite-sample results do not guarantee an
exact 5% rate or uniform p-values for every checkpoint. Some cases are conservative.

Power remains limited. Window tests detect twice-speed FullyNN futures in only
**5% / 3%** of trials for checkpoints 1/2, and twice-speed RMTPP in **27% / 36%**.
Quiet-window alternatives give FullyNN **28% / 26%**, RMTPP **100% / 100%**.
Stopped-quota tests detect suppression of all FullyNN stops in **100% / 100%**,
but suppression of RMTPP stops in only **8% / 11%**; forced immediate stops give
FullyNN **28% / 35%**, RMTPP **100% / 100%**. Non-rejection does not establish
equivalence, especially for rare finite events or rare natural stops. Improving
power needs a separately declared larger-sample or kernel experiment; thresholds
and model laws were not tuned after these results.

[Raw p-values, fitted parameter archives, before/after snapshots and provenance](validation/likelihood-exactness-20261014/provenance.json)
are retained. Both calibration scripts accept these frozen parameter JSON files
via `--checkpoint`, avoiding dependence on the expiration of CI checkpoint files.
The artifact ZIP digest was verified before extracting evidence.
