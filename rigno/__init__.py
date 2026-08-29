"""RIGNO package."""

import sys


if sys.version_info < (3, 11):
  raise RuntimeError(
    "RIGNO requires Python 3.11 or newer; "
    f"found Python {sys.version_info.major}.{sys.version_info.minor}."
  )
