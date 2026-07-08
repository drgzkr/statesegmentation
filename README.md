# statesegmentation with fast GSBS + nice plotting functions

This is a fork of the `statesegmentation` package with two additions:

1. **`fast_GSBS`** : an exact, drop-in accelerated version of `GSBS` (~5–10× faster, bit-identical results).
2. **Two plotting functions** : a fit-summary plot and a paper-ready time-by-time correlation matrix.

## Fast GSBS

`fast_GSBS` is a subclass of `GSBS` that replaces the compute-heavy internals with mathematically identical but much cheaper implementations. It returns **bit-identical** boundaries and t-distances to `GSBS` (verified across statewise detection on/off, every `finetune` mode, `dmin > 1`, and cross-validation via `y`). It is typically **~5–10× faster**, and because the optimization removes work that scales with the number of timepoints, the speedup **grows with the number of TRs** (the longest recordings benefit most; the number of voxels barely matters).

Use it exactly like `GSBS`, just change the class name:

```python
from statesegmentation import GSBS, fast_GSBS

gsbs_obj = fast_GSBS(kmax=x.shape[0] // 2, x=x)   # x is a (timepoints x voxels) array
gsbs_obj.fit()

gsbs_obj.bounds, gsbs_obj.states, gsbs_obj.strengths, gsbs_obj.nstates   # same API as GSBS
```

Because `fast_GSBS` subclasses `GSBS`, the plotting functions below (and everything else in `GSBS`) work on it too:

```python
gsbs_obj.plot_summary()
gsbs_obj.plot_time_by_time_corr_mtx()
```

A side-by-side speed/result comparison on placeholder data (400 voxels × 1200 TRs) is in [`examples/compare_gsbs.ipynb`](examples/compare_gsbs.ipynb). It fits both classes on identical data and checks that they agree.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drgzkr/statesegmentation/blob/main/examples/compare_gsbs.ipynb)

<details>
<summary>What changed under the hood (results are unchanged)</summary>

- **`_wdists` / `_wdists_state`** : the within-state score is computed from precomputed prefix sums of the data (O(V) per candidate boundary) and vectorized over all candidates, instead of re-z-scoring the whole timepoint×voxel matrix for every candidate.
- **`_tdist`** — the "same-state vs adjacent-state" pair masks are read from cached upper-triangle indices instead of rebuilding a full T×T `cdist` matrix every iteration.
- **`get_strengths`** : consecutive-state correlations via a single `einsum` instead of `scipy.stats.pearsonr` (whose p-value is never used).
- **Caching** — `cumsum(x)`, `cumsum(z)`, and per-state means are computed once per `fit()` instead of on every internal call.

`fast_GSBS` keeps its own copies of `fit()` and `_wdists_blocks()` because the base class dispatches those through the hard-coded class name; a plain subclass that only overrode the hot methods would silently fall back to the slow path. Keep those two in sync if the base algorithm ever changes.

Known limits: the t-distance builds a T×T covariance (O(T²) memory), which is fine at fMRI scale but caps very long recordings (e.g. native-rate EEG/MEG); and the 2-D statewise search is the main remaining cost.
</details>

## Plotting functions

Two new plotting functions (available on both `GSBS` and `fast_GSBS`):
- a summary plot which gives an overview of the GSBS fit, which can be plotted like this after fitting:
```python
gsbs_obj.plot_summary()
```
<img src="/readmeimages/GSBSsummaryPlot.png" width="500">

- a time by time correlation matrix figure with lots of setting for a paper-ready plot with fixed proportions etc, which can be plotted like this:
```python
gsbs_obj.plot_time_by_time_corr_mtx(
                                title = 'Global Brain Activity \n Time by Time Correlation Matrix',
                                scale = 0.9,
                                fontsize = 4,
                                line_color = 'black',
                                line_width = 0.5, # Width of the boundary lines
                                color_map = 'Spectral_r', # Colormap of the correlation matrix
                                tr_in_seconds = 2, # For the x and y axes labels in minutes
                                time_tick_fraction = 60, # This adjusts how dense the minute labels on the axes will be, values means every nth is shown
                                from_time = 0, # Timepoint
                                until_time = 200 # Timepoint
                                )
  ```
<img src="/readmeimages/GSBSCorrMtxPlot.png" width="500">

## Installation

This fork isn't on PyPI, so install it from GitHub:
```bash
git clone https://github.com/drgzkr/statesegmentation/
cd statesegmentation/
pip install .
```
Or install it directly (handy on Colab):
```bash
pip install git+https://github.com/drgzkr/statesegmentation.git
```

Here is the original readme of the package:

# statesegmentation

The statesegmentation package contains the implementation of a a greedy search algorithm (GSBS) to
segment a timeseries into states with stable activity patterns.
     
You can find more information about the method here:
Geerligs L., van Gerven M., Güçlü U. (2021) Detecting neural state transitions underlying event segmentation.
Neuroimage. https://doi.org/10.1016/j.neuroimage.2021.118085

The method has since been improved as described here:
Geerligs L., Gözükara D., Oetringer D., Campbell K., van Gerven M., Güçlü U. (2022)
A partially nested cortical hierarchy of neural states underlies event segmentation in the human brain.
BioRxiv. https://doi.org/10.1101/2021.02.05.429165

The package can be installed using: pip install statesegmentation

An example use case can be found in the examples directory.

