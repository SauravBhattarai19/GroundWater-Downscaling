"""
Coarse-to-Fine Downscaling Approach for GRACE Data

This package implements a scientifically rigorous downscaling method:
1. Upscale features from 5km to 55km (GRACE native resolution)
2. Train ML models at coarse scale (55km)
3. Apply models to fine resolution (5km)
4. Calculate and apply residual corrections

This approach respects the physics of GRACE measurements and prevents
artificial information injection.
"""

__version__ = "1.0.0"

