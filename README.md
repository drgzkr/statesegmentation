# statesegmentation with nice plotting functions

The exact package, except I added 2 new plotting functions:
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

Naturally, you can't install this fork with pip. Instead, you can do smt like this:
```bash
git clone https://github.com/drgzkr/statesegmentation/
cd statesegmentation/
pip install .
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


