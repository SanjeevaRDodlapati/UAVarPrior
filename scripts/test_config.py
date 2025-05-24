#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration testing script for UAVarPrior.

This script tests a configuration file by validating it and creating model components.
It's useful for ensuring that a configuration is valid before running a full training job.
"""
import os
import sys
import argparse
import logging
import yaml
import json
from pathlib import Path

# Add parent directory to path if running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from uavarprior.setup import load_path
from uavarprior.setup.run import validate_config, test_config_initialization
from uavarprior.utils import setup_logging

def test_config(config_path, output_path=None, verbose=False):
    """
    Test a configuration file
    
    Args:
        config_path: Path to configuration file
        output_path: Optional path to write test results
        verbose: Whether to print verbose information
        
    Returns:
        Tuple of (valid: bool, results: dict)
    """
    # Setup logging
    setup_logging(logging.DEBUG if verbose else logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    logger.info(f"Loading configuration from {config_path}")
    configs = load_path(config_path, instantiate=False)
    
    # Validate configuration
    logger.info("Validating configuration")
    valid, messages = validate_config(configs)
    
    if not valid:
        logger.error("Configuration is invalid:")
        for msg in messages:
            logger.error(f"  - {msg}")
        return False, {"valid": False, "messages": messages}
    
    logger.info("Configuration is valid:")
    for msg in messages:
        logger.info(f"  - {msg}")
    
    # Test initialization
    logger.info("Testing component initialization")
    init_results = test_config_initialization(configs)
    
    # Check if all initializations were successful
    all_success = all(result['success'] for result in init_results.values())
    
    if all_success:
        logger.info("All components initialized successfully")
    else:
        logger.warning("Some components failed to initialize:")
        for component, result in init_results.items():
            if not result['success']:
                logger.error(f"  - {component}: {result['error']}")
    
    # Combine results
    results = {
        "valid": valid,
        "messages": messages,
        "initialization": init_results,
        "all_success": all_success
    }
    
    # Write results to file if requested
    if output_path:
        logger.info(f"Writing results to {output_path}")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    return valid and all_success, results

def main():
    parser = argparse.ArgumentParser(description="Test configuration file")
    parser.add_argument("config_path", help="Path to configuration file")
    parser.add_argument("--output", "-o", help="Path to write test results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    success, _ = test_config(args.config_path, args.output, args.verbose)
    
    if success:
        print("Configuration test successful")
        sys.exit(0)
    else:
        print("Configuration test failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
