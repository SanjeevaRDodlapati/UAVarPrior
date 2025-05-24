#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model verification script for UAVarPrior.

This script verifies that a model can be properly loaded from a configuration file.
It attempts to create the model, loss function, and optimizer, and performs basic checks.
"""
import os
import sys
import argparse
import logging
import yaml
import torch
import importlib
from pathlib import Path

# Add parent directory to path if running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from uavarprior.model.factory import create_model
from uavarprior.setup import load_path
from uavarprior.utils import setup_logging

def verify_model(config_path, verbose=False):
    """
    Verify that a model can be properly loaded from a configuration file
    
    Args:
        config_path: Path to configuration file
        verbose: Whether to print verbose information
        
    Returns:
        True if successful, False otherwise
    """
    # Setup logging
    setup_logging(logging.DEBUG if verbose else logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {config_path}")
        configs = load_path(config_path, instantiate=False)
        
        # Check for model configuration
        if 'model' not in configs:
            logger.error("No 'model' section found in configuration")
            return False
        
        model_config = configs['model']
        
        # Check for required keys
        if 'class' not in model_config:
            logger.error("No 'class' key found in model configuration")
            return False
        
        # Create model
        logger.info("Creating model")
        model = create_model(model_config, train='train' in configs.get('ops', []))
        
        # Verify model structure
        logger.info(f"Model created successfully: {type(model).__name__}")
        logger.info(f"Backbone model: {type(model.backbone).__name__}")
        
        # Try forward pass with dummy input
        try:
            # Create dummy input based on model type
            # For sequence models, assume input is (batch_size, channels, seq_length)
            logger.info("Testing forward pass with dummy input")
            dummy_input = torch.randn(2, 4, 100)  # Batch of 2, 4 channels, length 100
            
            # Try forward pass
            with torch.no_grad():
                output = model(dummy_input)
            
            logger.info(f"Forward pass successful. Output shape: {output.shape}")
            
            # Check loss function
            if hasattr(model, 'loss_fn'):
                logger.info(f"Loss function: {type(model.loss_fn).__name__}")
                
                # Create dummy target
                if isinstance(model.loss_fn, torch.nn.MSELoss):
                    dummy_target = torch.randn_like(output)
                elif isinstance(model.loss_fn, torch.nn.BCEWithLogitsLoss):
                    dummy_target = torch.randint(0, 2, output.shape).float()
                else:
                    dummy_target = torch.randint(0, 2, output.shape).float()
                
                # Try loss computation
                try:
                    loss = model.loss_fn(output, dummy_target)
                    logger.info(f"Loss computation successful. Loss value: {loss.item()}")
                except Exception as e:
                    logger.error(f"Loss computation failed: {e}")
                    return False
            
            # Check optimizer
            if model.optimizer:
                logger.info(f"Optimizer: {type(model.optimizer).__name__}")
                
                # Try optimization step
                try:
                    # Forward pass
                    output = model(dummy_input)
                    loss = model.loss_fn(output, dummy_target)
                    
                    # Backward pass
                    model.optimizer.zero_grad()
                    loss.backward()
                    model.optimizer.step()
                    
                    logger.info("Optimization step successful")
                except Exception as e:
                    logger.error(f"Optimization step failed: {e}")
                    return False
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            return False
        
        logger.info("Model verification completed successfully")
        return True
    except Exception as e:
        logger.error(f"Model verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Verify model from configuration file")
    parser.add_argument("config_path", help="Path to configuration file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    success = verify_model(args.config_path, args.verbose)
    
    if success:
        print("Model verification successful")
        sys.exit(0)
    else:
        print("Model verification failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
