# statesegmentation with nice plotting functions

The exact package, except I added 2 new plotting functions:
- a summary plot which gives an overview of the GSBS fit.
<img src="/readmeimages/readme1.png" width="500">

- a time by time correlation matrix figure with lots of setting for a paper-ready plot with fixed proportions etc
<img src="/readmeimages/readme2.png" width="500">

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


