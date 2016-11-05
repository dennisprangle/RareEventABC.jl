# RE-ABC examples

This directory contains scripts used to perform all the analyses in the paper.
Some of these require a few extra packages.

## Normal example

* **iid_normal_comparison.jl** ABC, ABC-MCMC, MCMC and RE-ABC (adaptive and non-adaptive) analyses.

* **iid_normal_ll_dist.jl** Log-likelihood estimates and QQ plots under RE-ABC.

* **iid_normal_slice_cost.jl** Investigate number of slice sampling calls used by RE-SMC.

* **pmmh_normal_timed.jl** Compares the times taken by calls to RE-SMC in adaptive and non-adaptive RE-ABC.

## Epidemic example

* **Abakaliki_ABC.jl** ABC and ABC-MCMC analyses.

* **Abakaliki_adaptive_RE.jl** Adaptive RE-ABC analyses.

* **Abakaliki_RE.jl** Non-adaptive RE-ABC analyses.

* **pmmh_Abakaliki_timed.jl** Compares the times taken by calls to RE-SMC in adaptive and non-adaptive RE-ABC.




