"""diffusion_models.py
Public-facing imports for the diffusion model components used in this repo.

The original implementation lives in `modules_2.py` (legacy internal filename).
For clarity in a public GitHub release, new code should prefer importing from
this module, e.g.:

    from diffusion_models import GeneralDiffusionModel

This file re-exports the symbols from `modules_2.py` to preserve backwards
compatibility with existing scripts and paper-result reproductions.
"""

# Re-export all public symbols from the legacy module.
from modules_2 import *  # noqa: F401,F403

