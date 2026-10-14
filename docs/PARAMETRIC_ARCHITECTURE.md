# Parametric models and reproducibility

Branch: `codex/parametric-simulators`, based on `cae9eb7`.

## Validated results — 14 October 2026

Code `0dc9482` passes [Linux CI](https://github.com/NzoCs/Learning-point-processes/actions/runs/38552101343)
and [quality checks](https://github.com/NzoCs/Learning-point-processes/actions/runs/38552101339).
The unit suite has 267 passed, 6 skipped, 2 deselected and one expected failure
documenting FullyNN's known batch dependence. Coverage is 83.81%, above 80%.
All eleven historical train/test/predict pipelines pass; all nine neural
snapshots retain their historical values at the existing tolerances. All twelve
conditional pipelines pass. Hawkes, SelfCorrecting and Poisson reproduce exactly
the weights, simulated events, statistics, null draws and p-values after a
complete repeat, checkpoint prediction replay and epoch-boundary training resume.

The tiny integration fixture has eight sequences, two marks and only two
optimizer steps. It is a regression fixture, not a prediction-quality benchmark.

| Observable on that fixture | Before | After |
|---|---:|---:|
| Hawkes test NLL per event | 3.920192 | 3.920192 |
| Hawkes time MAE | 0.141250 | 48.965034 |
| Hawkes mean MMD | 16.321195 | 15.406283 |
| SelfCorrecting test NLL per event | 0.717339 | 0.715839 |
| SelfCorrecting time MAE | 0.129888 | 0.237720 |
| SelfCorrecting mark accuracy | 50% | 75% |
| SelfCorrecting mean MMD | 9.995576 | 11.800401 |

Hawkes' large MAE increase exposes long delays from a barely trained model;
the old generic sampler clipped those delays. This is not evidence that native
generation is less correct. SelfCorrecting's differences combine corrected
likelihood/intensity calculations, native prediction and changed random-number
consumption. Neither the better mark accuracy nor the changed MMD on eight
sequences establishes a scientific improvement. Both pooled historical p-values
remain 0.25 with only three null draws; that coarse fixture cannot assess calibration.

Independent fixed-horizon checks use 4096 trajectories per process and pass
against HawkesPyLib, NumPy and the SciPy inversion oracle. The independent marked
SelfCorrecting diagnostic additionally passes compensator rescaling and mark
checks. Seven neural constant-intensity cases and the IntensityFree lognormal
case pass their known-law diagnostics. These checks cover specified parameter
regimes, not all possible learned models or multivariate Hawkes settings.

[Raw before/after values, conditional controls, replay reports and provenance](validation/parametric-refactor-20261014/)
are archived from artifact `11887177894`, whose SHA256 was verified before reading.
CUDA was unavailable on these runners; CUDA tests are explicitly skipped.

## Architecture and observable behavior

`BaseModel` shares the Lightning contract, configuration and optimizer setup.
`TrainingStepsMixin` shares training, validation, test and prediction orchestration.
`NeuralModel` owns embeddings, Monte Carlo likelihood sampling and the generic
`EventSampler`. `ParametricModel` owns causal prefix prediction through dedicated
generation, without neural embeddings or generic thinning.

The historical eleven models are nine neural models (ANHN, ANHP, FullyNN,
IntensityFree, NHP, ODETPP, RMTPP, SAHP, THP), Hawkes and SelfCorrecting. Poisson
is additionally registered as a genuinely parametric model: the catalogue now
contains twelve models. A learnable parameter represented by `nn.Parameter`
does not make a process neural.

Historical prediction and test phases call `model.simulate`. Conditional
prediction continues to call it with only the observed prefix. Hawkes uses
multivariate Ogata thinning; SelfCorrecting uses exact competing-clock inversion;
Poisson uses exponential waits and categorical marks. Horizon-based Poisson
generation uses Poisson counts and sorted uniform times. Generation remains in
Torch on the model device. Fixed horizons use masks for variable-length results;
buffers grow rather than silently truncate events.

`PredictionSession` owns statistics and export lifecycle. The neural `Simulator`
adds generic generation. Parametric callbacks use the lighter session. The
existing `Simulator(parametric_model)` entry point delegates to native generation
as well, so it cannot accidentally restore the generic algorithm.

## Expected numerical changes

The neural historical reference files are deliberately unchanged and checked at
their existing tolerances. For Hawkes and SelfCorrecting, removing the unused
embedding changes random-number consumption, including training shuffle and
forecast draws. Historical test metrics and simulated paths consequently need
not match. Dedicated simulation removes generic proposal iteration limits and
delay clipping from these two models. Hawkes intensity queries at positive
delays now include excitation from the conditioning event, consistently with
the native generator. At zero delay the intensity API retains the pre-event
left limit for both Hawkes and SelfCorrecting.

SelfCorrecting additionally fixes two calculation errors: future intensities and
interval integrals include the event at the beginning of the interval; the event
likelihood still uses the intensity immediately before the event. The log-space
inverse (`logaddexp`, equivalent to `log1p` without intermediate exponential
overflow) and `expm1` retain small increments lost through ordinary log/exp
subtraction. Long histories no longer overflow the inverse through accumulated
penalties. Its loss, gradients, fitted parameters and simulations can therefore
change for mathematical reasons as well as RNG ordering.

Old parametric checkpoints contain an unused neural embedding and optimizer
state for that embedding. They are not interchangeable with the new architecture;
retrain from the saved dataset/configuration. Neural checkpoint parameter names
remain unchanged. Do not silently load old parametric optimizer state.

## Reproduce validation

From the repository root, with the locked development environment:

```bash
uv sync --frozen --no-default-groups --group dev --no-build-package pysiglib
uv run --frozen --no-sync python -m pytest tests -m 'not slow'
uv run --frozen --no-sync python -m scripts.architecture_regression --output artifacts/architecture-new
uv run --frozen --no-sync python -m scripts.independent_simulator_validation --size 4096 --output artifacts/references-new
```

Use a fresh output directory. The architecture suite runs train/test/predict on
all eleven historical models, compares the nine neural snapshots to reviewed
historical references, records parametric before/after values without replacing
their historical baselines, and runs the independent-data conditional pipeline
on all twelve catalogue models. The optional Linux workflow additionally checks
exact pipeline repetition, prediction replay and epoch-boundary training resume
for the three parametric models. Conditional controls check held-out futures,
sequence identifiers, unchanged checkpoints, exports and pooled p-values.

Independent distribution/timing comparisons are optional and require extra
reference tools; they are not runtime dependencies:

```bash
uv pip install --no-deps HawkesPyLib==0.3.0 numba==0.63.1 llvmlite==0.46.0
uv run --frozen --no-sync python -m scripts.parametric_benchmark --samples 4096 --repeats 7 --output artifacts/benchmark-new
```

Hawkes uses HawkesPyLib's univariate exponential process, with
`eta=alpha` and `theta=1/beta`. Poisson uses independent NumPy count/uniform
generation. SelfCorrecting uses numerical compensator inversion with SciPy
`brentq` and NumPy exponential draws: this is an independent numerical oracle,
not a ready-made SelfCorrecting library. Comparisons use the same empty history,
parameters and horizon, without selecting trajectories by their event count.
Counts and first-event distributions are checked across independent random streams.

The optional GitHub workflow `parametric-refactor.yml` runs these controls on
Linux CPU. It can be dispatched manually or requested by the commit marker
`[parametric-refactor]` on this branch. Raw reports and pipeline artifacts are
retained. CUDA tests explicitly skip without hardware; `--device cuda` refuses
to substitute CPU. This refactor introduces neither JAX nor a new execution backend.

## Remaining limits

Historical regression success establishes stability, not scientific correctness
of every neural model. Independent diagnostics still expose FullyNN's intensity
dependence on batch/sample size and its bounded cumulative-hazard parameterization.
The generic neural sampler still has proposal iteration limits and delay
clipping. These paths are not changed or hidden by this parametric refactor.

The likelihood observation contracts also differ: Hawkes scores every valid
event over `[0,T]`, whereas SelfCorrecting and Poisson condition on the first
event and score subsequent events. This historical distinction is preserved;
their reported NLLs must not be used directly to rank models without aligning
the observation window and conditioning contract first.

Local and ordinary hosted Linux CPU checks do not validate CUDA, DDP or execution
on Ruche. GPU behavior, reproducibility across devices and GPU execution times
must be measured on appropriate hardware. Equal seeds across different devices
or libraries do not imply identical random trajectories.
