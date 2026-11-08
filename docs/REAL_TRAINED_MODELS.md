# Actual trained models: real-data fitting and conditional model tests

The initial pilot trains NHP/RMTPP on the pinned volcano/retweet datasets: four
new fits, hidden16, batch32, at most30epochs, early stopping patience3, seed202611051.
Only validation loss selects the checkpoint. Every epoch can checkpoint; no
evaluation p-value selects weights. The public RunnerManager performs fitting
and native held-out prediction tests. Existing checkpoints are inventoried but
none match these four dataset/model configurations; no historical weights are
converted, overwritten or retrained as artifact repair.

Time units are fixed from training only: divide every split by eight times the
median positive training gap. Marks, order, first real event and rounded-time
ties are preserved. No observation end is invented. Source revisions, split
fingerprints, full snapshot hashes and the divisor are recorded. Published
splits do not certify independent physical subjects; real-data prediction metrics
are descriptive and are not calibrated fixed-window goodness-of-fit tests.

Each checkpoint also defines a conditional generative target given one common
four-event history selected from training, never from test. The public Simulator
generates independent full windows T=1. A window is accepted only when a sampled
crossing event or native certified infinite wait proves complete exposure.
An event budget is expanded from32 to4096 with the same seed; exhaustion,
envelope violations, duplicate/invalid times or other numerical failures stop,
without truncating a tail or filtering trajectories. Candidate-bound safety
checks do not establish an analytic global envelope for every neural intensity;
simulator fidelity remains part of the diagnostic scope.

H0 compares two independent draws from that fixed trained generator. The known
alternative compresses future time by0.8, first simulating through1/0.8. The
reference history and fitted weights are fixed for all tests/permutations.
The alternative pool is passed first as the observed pool; this matters for
one-sample Stein, while the two-sample MMD/loss comparisons are symmetric.
Seven methods share the same trajectories: signature RBF counting/interval/event
paths, linear counting signature, count RBF, relative squared prediction loss
and Stein–Papangelou. These answer distinct hypotheses. KSD uses the fixed model
density; MMD calibrates equality of generator laws, including possible simulator
approximation. The neural adapter always conditions on the real fixed history.

Each method has an independent capability/cost sub-study at n=16 then128 with
60seconds per comparison. An unsupported or too-expensive method has an explicit
failure record and no p-value. Other predeclared sub-studies can proceed; the
overall report remains partial. A failure during Monte Carlo stops that cell,
with no failed trial discarded. Supported methods use100 fresh H0/100 fresh
time-compression pairs, n=128, shared whole-trajectory swaps499 (KSDbootstrap999).
Exact intervals condition on the fixed checkpoint/history; H0 excess family28
and Holm84 are fixed in advance. Unavailable comparisons are not fabricated.

All training, data preparation and remote tests are explicit Slurm operations,
outside CI. The four training tasks follow the active Hawkes GPU array so total
GPU concurrency stays at most four. Synthetic model tests follow the training
array and verify each checkpoint hash. The final CPU audit replays all source
hashes/p-values and preserves unavailable cases. No real-data distributional
adequacy, universal power or general retraining uncertainty is claimed.
