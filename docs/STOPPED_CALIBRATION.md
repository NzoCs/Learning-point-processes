# Calibration of terminating learned processes

This experiment freezes the corrected FullyNN and RMTPP weights archived in
`docs/validation/unbounded-sampling-20261014/*_snapshot.json`. These models were
fitted before this experiment; none of the sampled evaluation futures is used
to select or fit their parameters. The checkpoints are small integration fits,
not claimed to estimate a real-world process accurately.

For each model, 500 H0 trials compare the production simulator with a separately
implemented NumPy/SciPy continuation generator. The frozen Torch history encoder
is shared; marked intensities and their analytic derivatives, integrated hazards,
limits at infinity, NumPy random draws and SciPy inverse roots are independent
of the production sampler. This validates conditional p-values against an
independent elapsed-time implementation, not the encoder against a separate
implementation or uncertainty from fitting the model to the evaluation data.

Each trial has independent batches of 16, 11 and 5 matched heterogeneous histories,
two event marks and a continuation budget of six events. A continuation either
reaches the budget or terminates; padding is never an event. Sequences with zero
future events remain in the test. Actual termination is explicitly recorded,
and is distinguished from an administrative event-count censoring at the budget.

Two observation protocols are compared on exactly the same data and random swaps:

* The historical conditional protocol truncates each pair to the smaller last
  event time. A known empty terminal future is now accepted, but truncation can
  erase its difference from a nonempty future by making both horizons zero.
* `stopped_event_budget=6` explicitly selects `complete_stopped_event_budget_v1`.
  It keeps the complete observed futures and multiplies the signature kernel
  by a unit-bandwidth Gaussian kernel on the boolean terminal flag. The same
  history kernel and paired off-diagonal U-statistic are retained. Full pooled
  normalization and terminal flags commute with within-history pair swaps, so
  the complete swap orbit is fixed. This mode requires explicit matching
  termination flags and a common event budget for both groups.

The model definitions, prior observation defaults, p-value formula, historical
baselines and old nonterminating observation contracts are not replaced.
The ordinary fixed-event holdout loader still requires enough observed events;
the stopped mode accepts independently prepared histories and terminal futures
through the public test API. It is not automatically selected by that loader.

```python
test = ConditionalMMDTwoSampleTest(kernel, 99, stopped_event_budget=6)
test.compute_statistics(observed, simulated, histories=histories, pair_ids=ids)
```

There are 99 independent within-pair randomizations per batch. Aggregate null
draws use the product group across disjoint batches. A cached Gram calculation
is checked against every public-API statistic and null draw on the first trial
of every case/batch. Reported p-values use the existing plus-one correction and
conservative numerical ties; no bandwidth is fitted to rejection results.

Before inspecting the large-run results, the H0 diagnostic is fixed as a
one-sided exact binomial excess test against 5%, with Bonferroni significance
0.01/8 over two models, two protocols and two aggregation methods. Raw p-values,
Wilson 95% intervals, fingerprints of both samples including termination flags,
empty/terminal row counts and checkpoint hashes are retained. Absence of detected
excess is finite-sample evidence, not a proof of validity for every checkpoint.
Strong conservatism is reported separately rather than called nominal calibration.

Two predeclared alternatives have 200 trials each: every next-event draw is
conditioned on existence (`never_stop`), or every continuation immediately
terminates (`always_stop`). Power is reported without post-hoc tuning or a
claimed guarantee against small differences. The historical protocol's loss
of information is a diagnostic control, not a reason to exclude terminal rows.

```bash
uv run --frozen --no-sync python -m scripts.stopped_conditional_calibration \
  --model FullyNN --trials 500 --power-trials 200 --permutations 99 \
  --output artifacts/stopped-FullyNN-new
```

Repeat for RMTPP with a fresh output directory. The optional Linux CPU workflow
`stopped-calibration.yml` runs both models only on explicit dispatch or a branch
push containing `[stopped-calibration]`. It does not require Ruche or GPU.

## Results, 14 October 2026

Commit `edc799abd014027c2c972abfde773bdfb2684c58` passed the
[complete Linux workflow](https://github.com/NzoCs/Learning-point-processes/actions/runs/38581548433)
and [lint](https://github.com/NzoCs/Learning-point-processes/actions/runs/38581548223).
The main suite has 301 passing tests, 7 skips, 2 deselections and 84.21% coverage.
Both calibration jobs and frozen-parameter checks passed. Archive digests are
verified and [raw p-values/provenance](validation/stopped-calibration-20261014/provenance.json)
are retained without replacing any historical reference.

Complete stopped protocol, 500 H0 trials per model, nominal alpha=5%:

| Model | First batch false positives | Aggregated false positives | Aggregated Wilson 95% interval |
| --- | ---: | ---: | ---: |
| FullyNN | 11/500 (2.2%) | 15/500 (3.0%) | 1.83–4.89% |
| RMTPP | 33/500 (6.6%) | 26/500 (5.2%) | 3.57–7.51% |

No H0 excess is detected by the predeclared criterion. FullyNN is conservative,
not nominally uniform: many terminal empty paths produce ties and small batches
have few informative pairs. RMTPP's intervals are compatible with 5%; the first
batch's 6.6% is retained rather than discarded. These finite-sample results do
not certify arbitrary checkpoints, kernels, history distributions or horizons.

Across H0 there are 16000 observed and 16000 simulated rows per model. FullyNN
has 13045/12975 empty rows and 15999/16000 terminal rows (observed/simulated).
RMTPP has 446/471 empty rows and 2441/2372 terminal rows. Empty paths are retained,
not filtered or replaced with a finite next event.

Aggregated power, 200 trials per alternative:

| Model | Protocol | Suppress every natural stop | Stop every future immediately |
| --- | --- | ---: | ---: |
| FullyNN | Complete stopped | 200/200 (100%) | 55/200 (27.5%) |
| FullyNN | Historical truncation | 10/200 (5%) | 0/200 (0%) |
| RMTPP | Complete stopped | 14/200 (7%) | 200/200 (100%) |
| RMTPP | Historical truncation | 14/200 (7%) | 0/200 (0%) |

The historical protocol is not a suitable stopping-probability diagnostic: it
erases every immediately stopped alternative. The explicit complete mode keeps
the missing-event information. Its power is still limited for RMTPP suppression
of relatively rare natural stops and FullyNN's additional early stops; absence
of rejection must not be interpreted as equivalence. Larger batches would be a
separate predeclared power experiment, not an after-the-fact threshold change.

This event-budget design requires a genuinely known terminal status. No events
within a finite observation window alone cannot establish permanent termination.
Data collected over a time window need a matched, predeclared horizon and a
separate calibration of that observation design. The production fixed-event
holdout loader is not silently converted to the stopped experiment's design.
