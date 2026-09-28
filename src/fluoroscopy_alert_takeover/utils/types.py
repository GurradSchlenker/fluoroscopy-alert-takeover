"""Array aliases shared by every module.

Ref: none - release-internal typing only.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]
StrArray = npt.NDArray[np.str_]

__all__ = ["BoolArray", "FloatArray", "IntArray", "StrArray"]
