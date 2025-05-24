"""Helper functions for importing PeakGVarEvaluator"""

import importlib
import logging

logger = logging.getLogger(__name__)

def try_import_peakgvarevaluator(analyze_seqs_info):
    """
    Try different import strategies to load PeakGVarEvaluator.
    """
    # Try multiple import strategies in order
    import_paths = [
        "src.uavarprior.predict.seq_ana.gve",  # Most specific path first
        "uavarprior.predict.seq_ana.gve",
        "src.uavarprior.predict.seq_ana",      # Try seq_ana package
        "uavarprior.predict.seq_ana",
        "src.uavarprior.predict",              # Legacy paths as fallback
        "uavarprior.predict"
    ]
    
    for import_path in import_paths:
        try:
            logger.debug(f"Trying to import PeakGVarEvaluator from {import_path}")
            module = importlib.import_module(import_path)
            if hasattr(module, "PeakGVarEvaluator"):
                logger.info(f"Found PeakGVarEvaluator in {import_path}")
                return module.PeakGVarEvaluator(**analyze_seqs_info)
        except ImportError as e:
            logger.debug(f"Could not import module {import_path}: {e}")
            continue
    
    # If all paths fail, try direct class paths
    class_paths = [
        "src.uavarprior.predict.seq_ana.gve.PeakGVarEvaluator",
        "uavarprior.predict.seq_ana.gve.PeakGVarEvaluator"
    ]
    for path in class_paths:
        try:
            logger.debug(f"Attempting direct class import from {path}")
            module_path, class_name = path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            logger.info(f"Successfully imported PeakGVarEvaluator from {path}")
            return cls(**analyze_seqs_info)
        except Exception as e:
            logger.debug(f"Direct class import failed from {path}: {e}")
    
    return None
