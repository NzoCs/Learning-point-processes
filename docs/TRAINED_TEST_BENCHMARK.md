# Trained models and statistical tests

This branch adds public, fixed-exposure test APIs. It does not publish new power
results, execute training, download datasets, submit Slurm jobs or change models.
The existing stopped/conditional MMD API retains its explicit observation design.
Do not compare its p-values with fixed-window tests while silently changing the
observed portion of the trajectory.

## What each test answers

| Method | Inputs | Null and limitations |
|---|---|---|
| Fixed-window signature MMD | Two independent matched pools; same complete exposure | Exchangeable trajectory pairs; count or exact-gap paths; full pooled Gram |
| Relative prediction loss | Two pools and one common frozen reference | Difference in mean relative squared error; not a characteristic distribution test; no square root |
| Stein–Papangelou | Observed pool and one fixed density/intensity model | Adequacy to that model; centered multinomial bootstrap is asymptotic; histogram-RBF kernel is not claimed characteristic |

All tests return the observed statistic, every null statistic, p-value, null
method and null sample count. Do not average p-values or MMDs over blocks.

## Public API

```python
from new_ltpp.configs.window_test_config import WindowTestConfig
from new_ltpp.evaluation.statistical_testing.references import NeuralIntensityReference
from new_ltpp.evaluation.statistical_testing.statistical_tests import create_statistical_test

# Load the trained model through the current public ModelFactory and call eval().
# observed and simulated are public Batch objects with known exposure T.
reference = NeuralIntensityReference(model.eval())
cfg = WindowTestConfig(test_type="ksd", horizon=T, num_event_types=K,
                       stein_bins=8, n_samples=999, seed=202611051)
test = create_statistical_test(cfg, reference=reference)
result = test.compute_statistics(observed)

cfg = WindowTestConfig(test_type="window_mmd", horizon=T,
                       num_event_types=K, representation="intervals")
result = create_statistical_test(cfg).compute_statistics(observed, simulated)
```

`PoissonReference` and `HawkesReference` provide known-model controls. Hawkes
conditions on empty past: use matched burn-zero data for nominal calibration.
Hidden-burn archives are diagnostics of this working model, not exact H0 controls.

The prediction loss averages `((target - mean)/mean)**2` over zero and every
event origin, including the censored final interval. It then compares equally
weighted path scores by whole-trajectory paired swaps. Means are conditional
restricted expectations, not uncensored mean intervals. The same frozen model
and parameters are used for both groups and every permutation.

The interval path contains `(gap_index/32, log1p(gap/0.02))` and marked one-hot
coordinates when applicable. It retains all complete gaps and the terminal
censored gap, splits segments into three and pads with a repeated endpoint.
The counting path uses raw physical time and cumulative counts divided by an
explicit fixed scale. These settings are not estimated from evaluation labels.
The counting path supports the library static linear and RBF kernels. A
count-only RBF baseline also runs through the same paired MMD test.
Neither embedding is the existing normalized, stopped-event `SIGKernel` path.
Both use the library's sole `PySigLibKernel` backend; there is no script bypass.

## Model capability and conditioning

The neural reference evaluates the **public causal intensity interface**, using
the checkpoint's native dtype. It integrates the full horizon including terminal
survival. Insertion ratios recompute all later histories and intensities. It does
not treat training loss, a gradient with respect to weights, or past-only
intensity as a Papangelou intensity.

Empty-history NHP and fixed-history NHP/THP/SAHP/RMTPP are tested against
constant-intensity controls; stored trained NHP weights are also tested.
Some architectures have no unconditional empty-history intensity in the present
API. They raise an explicit error; no invented BOS event or prediction-changing
architecture patch is applied. A real common conditioning history can be supplied
as `history_times=[0., .2], history_marks=[0, 1]`; observed future times are then
relative to `.2`. The target is the conditional law given that fixed history.
This does not implement arbitrary distinct histories for every observed row.
Such model/test combinations stay unsupported until that contract is implemented.

## Prepared experiment matrix

See [the machine-readable plan](../configs/experiments/trained_test_benchmark.json).
Start with NHP/RMTPP, volcano/retweet and one training seed. These dataset names
are registered candidates, not evidence that local snapshots/checkpoints exist.
Extend to SAHP, taxi and Stack Overflow with three training seeds only after
loading, numerical, observation and runtime gates pass. Training and validation
select checkpoints; evaluation p-values never select checkpoints or kernels.

Separate oracle calibration, fresh simulation from each trained fixed model,
controlled alternatives, and held-out real-data model criticism. Preserve marks
and independent trajectory/subject IDs. Fix units, windows and tuning from the
training split. Do not shift the first observed event to zero and claim an
unconditional observation. Do not select windows based on future event counts.

Controlled alternatives include count rate, excitation, exponential vs power p3,
temporal shuffling at fixed counts, mark probabilities, time/mark dependence,
and the same-relative-loss counterexample. All methods share each observation
pool and its simulation. Never call real-data rejection frequency "power" without
a known alternative or "miscalibration" without a genuine H0 generator.

Start at n=128. A 16-path capability probe precedes the pilot. Stop if a full pool
exceeds 60 seconds, numerical checks fail or the chosen Stein state budget is
exceeded; do not silently change quadrature, marks, sample size or test settings.
The initial 100 trials per cell are a pilot, not a precise 5% calibration study.
Extend independent Monte Carlo repetitions after cost and statistical review.
Report exact intervals per training seed; paired McNemar comparisons use a
predeclared Holm family. Repeated sampling of one held-out real dataset measures
stability, not independent binomial power. Record effect size and GPU peak memory
in addition to rejection rate and wall time.

## Artifact preflight and execution

`python -m scripts.inventory_test_artifacts ROOT... --output inventory.json`
inventories all local weights under the supplied roots without unpickling them.
It includes stored JSON trained-state fixtures; data artifacts are listed
separately. Complete the remote inventory when Ruche is accessible. Existing
class paths, weights, checkpoint schemas and predictions are unchanged by this
evaluation-only change. No artifact migration is needed for these additions.
If a stored checkpoint cannot load through the **current** factory and strict
state loading, stop and prepare
a separate versioned migration for every affected artifact and consumer; do not
add historical aliases, wrappers, shape conversions or automatic retraining.

`python -m scripts.trained_test_benchmark MANIFEST.json` performs read-only
artifact preflight. It never loads a pickle checkpoint or executes a test.
`--execute` explicitly loads a **trusted** current checkpoint by initializing
`ModelFactory` from its canonical `model_config`/`data_info` and strictly loading
`state_dict`. It runs the public test suite, refusing output overwrites and CUDA
fallback. This is a single-pool
evaluation command; it does not train or orchestrate the full experiment matrix.

The JSON manifest requires exactly: `model_id`, `checkpoint`, `checkpoint_sha256`,
`observed`, `observed_sha256`, `simulated`, `simulated_sha256`, `training_ids`,
`training_ids_sha256`, `simulation_checkpoint_sha256`, `observation_protocol`,
`tests`, `device`, `output`. Relative input paths resolve against the manifest.
The observation protocol is `independent_fixed_windows_v1`; `tests` contains
complete `WindowTestConfig` dictionaries with common horizon and mark count.
`device` is `cpu` or `cuda`. `training_ids` is a JSON list of all units used to
fit or select the model, including validation units.

Each immutable NPZ pool contains `time_seqs`, integer `type_seqs`, boolean
`valid_event_mask`, string `sequence_ids`, scalar `horizon`, and per-row
`coverage_end` at least the horizon. This last field certifies observed exposure,
not the last event; a stopped simulation must have reached the horizon or have
an explicit no-future-event guarantee. It cannot claim an unobserved tail empty.
Pools must have matching sizes and horizons; independent unit IDs must be unique,
disjoint from fitting/selection, and distinct between observed and simulated.
The simulation checkpoint hash must match the evaluated checkpoint.

## References and interpretation

Yang, Rao and Neville (AISTATS 2019), [Stein–Papangelou](https://proceedings.mlr.press/v89/yang19a.html),
equations 2, 7, 12–14 motivate the insertion operator and bootstrap. The reference
model supplies extra information relative to sample-only MMD. The kernel still
matters: the paper does not imply universal superiority, nor does our finite
histogram representation have a guarantee to detect every alternative.
Report unsupported cases, all numerical failures, training variability and
computational costs. Do not hide errors by excluding trials.

## Explicit Ruche pilot

`scripts.ruche_trained_test_pilot` implements the first execution stages using
the frozen `ruche_trained_pilot_20261108.json` plan. Its two oracle cells use the
library Poisson generator on complete windows: a rate change and a mark-law
change with identical total rate. Six methods share each pool, with 100 H0 and
100 H1 repetitions per cell after 16/128-path cost gates. All trajectories and
null vectors are retained. Warm-up has a separate 300-second allowance; an
evaluated full pool must stay within 60 seconds.

The existing trained NHP fixture is a separate CUDA capability/cost probe, not
an experiment on a newly trained volcano/retweet model and not a calibration
study. The dataset audit pins the registered revisions and records the available
schema and split sizes. Its outputs require explicit exposure and independent
unit certification before the real-data training/evaluation matrix can run.
These stages never invent exposure from the last event. Cluster submissions are
performed separately with receipts; importing this module submits nothing.
