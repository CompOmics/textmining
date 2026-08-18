"""
Reproducibility utilities for the PRIDE Metadata Extraction Framework.

This module provides seed management and reproducibility controls for
ensuring consistent results across runs.
"""

import os
import random
import logging

logger = logging.getLogger(__name__)

# Global seed state
_GLOBAL_SEED = None


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    This function sets seeds for:
    - Python's random module
    - NumPy (if available)
    - PyTorch (if available)
    - Environment variable for hash randomization
    
    Args:
        seed: Integer seed value. Use None to disable seeding.
    """
    global _GLOBAL_SEED
    
    if seed is None:
        logger.info("Seed is None - running in non-deterministic mode")
        return
    
    _GLOBAL_SEED = seed
    logger.info(f"Setting global seed: {seed}")
    
    # Python random
    random.seed(seed)
    
    # Hash seed for reproducible dict ordering
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # NumPy
    try:
        import numpy as np
        np.random.seed(seed)
        logger.debug("NumPy seed set")
    except ImportError:
        pass
    
    # PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # For complete reproducibility (may impact performance)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.debug("PyTorch seed set")
    except ImportError:
        pass


def get_seed() -> int:
    """Get the current global seed value."""
    return _GLOBAL_SEED


def get_reproducibility_info() -> dict:
    """
    Get information about the current reproducibility state.
    
    Returns:
        Dictionary with seed and library version information.
    """
    info = {
        "seed": _GLOBAL_SEED,
        "python_hash_seed": os.environ.get('PYTHONHASHSEED', 'not set'),
    }
    
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cudnn_deterministic"] = torch.backends.cudnn.deterministic
    except ImportError:
        info["torch_version"] = "not installed"
    
    try:
        import numpy as np
        info["numpy_version"] = np.__version__
    except ImportError:
        info["numpy_version"] = "not installed"
    
    try:
        import transformers
        info["transformers_version"] = transformers.__version__
    except ImportError:
        info["transformers_version"] = "not installed"
    
    return info
