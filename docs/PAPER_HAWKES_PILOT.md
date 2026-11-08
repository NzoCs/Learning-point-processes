# Convergent-kernel Hawkes experiment

This is a new finite-start pilot, separate from all historical archives. The
kernel is `n_star*b*(1+b*t/(p-1))**(-p)`, with b=10, n_star=.5 and mu=50. Its
integral is .5 for every p>1; the stationary rate target is 100. The implemented
Poisson-cluster generator retains every ancestor from -B, including descendants
that enter [0,1]. It uses no thinning envelope and fails on resource overflow.
It does not claim exact stationary initialization.

The frozen plan compares the exponential kernel and p=1.5,2,3,5,8,16,32.
Burn sensitivity uses B=10,50,100,500 with two independent groups of 16 paths per
family/B. Raw visible and prehistory events, counts, gaps, ten subwindow counts
and independent-unit intervals are saved. A kernel tail mass is a diagnostic,
not a stationary-error bound including descendants.

GPU cost probes use the same n=128 pools for grids 25/50/100/200/500, event paths,
interval paths, count-only RBF, linear signature grid100, and common-reference
relative prediction loss. Count channels use count/10, time is physical, RBF
sigma=.3, gap scale=.02 and rank scale=32. Settings are fixed before evaluation.
Event paths alternate time movement and count jumps at a fixed time, with final
endpoint padding. The signature solver remains finite differences at order0;
same-path tiny controls also record orders1/2. No injectivity/exactness claim is
made. The three-point grid collision in the handoff is tested explicitly.

The loss uses a common frozen exponential Hawkes empty-past working predictor,
not the true oracle prediction given the hidden burn history. Its permutations
still exchange whole independent trajectories. KSD is not applied to this
hidden-past target: the available density adapter would otherwise use the wrong
initial law. The separate Poisson pilot validates that method under exact H0.

Only after the burn/cost stages complete, queued evaluation jobs check all ten
methods and the <=60-second full-suite cost gate. Failures stop evaluation and
are preserved, with no method dropped or settings changed. B=500 is fixed for
these fresh independent draws, regardless of pilot p-values; conclusions remain
about finite-start B=500 and must be read alongside burn diagnostics. The phase
uses 100 fresh bank pairs per case, independent data and permutation streams,
499 paired whole-trajectory swaps and alpha=.05. This is a pilot, not a precise
calibration study. Ten H0 excess controls use .01/10; power comparisons use exact
McNemar with Holm315 and binomial intervals over independent trial units.

The final CPU audit replays every p-value and input-pool hash, requires all eight
cases, and emits JSON/CSV before PDF/PNG power plots. Partial cells never become
final power estimates. All source, Slurm and environment receipts live in a
separate dated campaign archive.

The exact mean-feature counterexample is separate: inclusion coefficients cancel
for P={{1,2},{3,4}} and Q={{1,3},{2,4}}. This concerns the linear inner-product
kernel on mean features. It does not prove blindness of the current library
`MKernel`, which transforms a distance nonlinearly. Historical checkpoint reuse
and systematic neural extinction studies remain separate later work.
