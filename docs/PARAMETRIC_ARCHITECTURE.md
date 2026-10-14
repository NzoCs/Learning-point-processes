# Parametric models and reproducibility

Branch: `codex/parametric-simulators`, based on `cae9eb7`.

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
likelihood still uses the intensity immediately before the event. `log1p` and
`expm1` retain small increments that ordinary log/exp subtraction loses in
float32. Its loss, gradients, fitted parameters and simulations can therefore
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
on all twelve catalogue models. Conditional controls check held-out futures,
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

Local and ordinary hosted Linux CPU checks do not validate CUDA, DDP or execution
on Ruche. GPU behavior, reproducibility across devices and GPU execution times
must be measured on appropriate hardware. Equal seeds across different devices
or libraries do not imply identical random trajectories.
