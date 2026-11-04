"""Load PyTorch before any test imports LightGBM.

Both link an OpenMP runtime on macOS. One process survives both only when PyTorch's loads
first; the reverse order crashes with a segmentation fault once LightGBM or PyTorch runs
parallel code. The fitting tests still run LightGBM in a clean interpreter (test_lightgbm_cv).
"""

import torch  # noqa: F401
