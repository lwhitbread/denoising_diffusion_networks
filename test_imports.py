#!/usr/bin/env python
"""
test_imports.py
Quick script to verify all dependencies are correctly installed.
Run this after creating the conda environment to check everything is working.
"""

def test_imports():
    print("Testing imports...\n")
    
    # Core Python
    print("✓ Testing standard library imports...")
    import os
    import sys
    import json
    import argparse
    import math
    import random
    import multiprocessing as mp
    from typing import List, Tuple, Dict, Optional, Any
    print("  Standard library OK")
    
    # Scientific computing
    print("\n✓ Testing scientific computing packages...")
    import numpy as np
    print(f"  numpy {np.__version__}")
    
    import pandas as pd
    print(f"  pandas {pd.__version__}")
    
    import scipy
    print(f"  scipy {scipy.__version__}")
    
    import sklearn
    print(f"  scikit-learn {sklearn.__version__}")
    
    import statsmodels
    print(f"  statsmodels {statsmodels.__version__}")
    
    # PyTorch
    print("\n✓ Testing PyTorch...")
    import torch
    print(f"  PyTorch {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU count: {torch.cuda.device_count()}")
    
    # Matplotlib
    print("\n✓ Testing visualization packages...")
    import matplotlib
    print(f"  matplotlib {matplotlib.__version__}")
    
    # Optional but recommended
    print("\n✓ Testing optional packages...")
    try:
        from tqdm import tqdm
        print("  tqdm OK")
    except ImportError:
        print("  tqdm not found (optional)")
    
    # R and rpy2 (for GAMLSS)
    print("\n✓ Testing R and rpy2...")
    try:
        import rpy2
        print(f"  rpy2 {rpy2.__version__}")
        
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr
        
        r_base = importr("base")
        print(f"  R version: {r_base.version[12][0]}")
        
        try:
            gamlss = importr("gamlss")
            gamlss_dist = importr("gamlss.dist")
            print("  gamlss and gamlss.dist packages OK")
        except Exception as e:
            print(f"  WARNING: GAMLSS packages not found: {e}")
            print("  GAMLSS baseline will not be available")
    
    except ImportError as e:
        print(f"  WARNING: rpy2 not available: {e}")
        print("  GAMLSS baseline will not be available")
    
    # Project-specific modules
    print("\n✓ Testing project modules...")
    try:
        from diffusion_models import GeneralDiffusionModel
        print("  diffusion_models OK")

        # Legacy import path kept for backwards compatibility
        from modules_2 import GeneralDiffusionModel as _LegacyGeneralDiffusionModel  # noqa: F401
        print("  modules_2 (legacy) OK")
        
        from train_diffusion import train_diffusion_model
        print("  train_diffusion OK")
        
        from train_diffusion_saint import build_saint_diffusion_model
        print("  train_diffusion_saint OK")
        
        from saint_tabular import SAINTDiffusionModel
        print("  saint_tabular OK")
        
        try:
            from gamlss_adapter import GAMLSSBaseline
            print("  gamlss_adapter OK")
        except ImportError:
            print("  gamlss_adapter not importable (requires R/rpy2)")
    
    except ImportError as e:
        print(f"  ERROR: Project module import failed: {e}")
        print("  Make sure you're running this from the bundle directory")
        return False
    
    print("\n" + "="*60)
    print("✓ All core dependencies are correctly installed!")
    print("="*60)
    return True


if __name__ == "__main__":
    import sys
    success = test_imports()
    sys.exit(0 if success else 1)

