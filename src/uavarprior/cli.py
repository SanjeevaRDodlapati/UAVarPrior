"""Command line interface for UAVarPrior.

Why does this file exist, and why not put this in ``__main__``? You might be tempted to import things from ``__main__``
later, but that will cause problems--the code will get executed twice:

- When you run ``python3 -m uavarprior`` python will execute``__main__.py`` as a script. That means there won't be any
  ``uavarprior.__main__`` in ``sys.modules``.
- When you import __main__ it will get executed again (as a module) because
  there's no ``uavarprior.__main__`` in ``sys.modules``.

.. seealso:: http://click.pocoo.org/5/setuptools/#setuptools-integration
"""
import os
import click
import logging
import sys
from typing import Optional

# Add the parent directory to sys.path to allow direct imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from uavarprior import __version__
    from uavarprior.setup import load_path, parse_configs_and_run
    from uavarprior.utils import setup_logging
except ImportError:
    # Fallback to local imports if package imports fail
    from uavarprior.utils.logging import setup_logging
    from uavarprior.setup import load_path, parse_configs_and_run
    __version__ = "0.1.0"  # Default version if not available

@click.group()
@click.version_option(__version__)
@click.option('--verbose', '-v', count=True, help='Increase verbosity (can be used multiple times)')
@click.option('--cuda-blocking', is_flag=True, help='Enable CUDA launch blocking for debugging')
def cli(verbose: int, cuda_blocking: bool):
    """UAVarPrior - Uncertainty-Aware Variational Prior for [task description]."""
    # Setup logging based on verbosity
    log_level = max(logging.WARNING - verbose*10, logging.DEBUG)
    setup_logging(log_level)
    
    if cuda_blocking:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

@cli.command()
@click.argument('config_path', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option('--override', '-o', multiple=True, help='Override config parameters: -o section.param=value')
@click.option('--debug', is_flag=True, help='Enable debug mode with detailed error tracking')
def run(config_path: str, override: Optional[tuple], debug: bool):
    """Run model training or inference using the specified configuration file."""
    try:
        logging.info(f"Loading configuration from {config_path}")
        configs = load_path(config_path, instantiate=False)
        
        # Basic validation of the loaded configuration
        if not isinstance(configs, dict):
            raise ValueError(f"Configuration from {config_path} did not load as a dictionary")
        
        logging.debug(f"Configuration loaded with keys: {list(configs.keys())}")
        
        # Add debugging output to see the config structure
        logger = logging.getLogger("uavarprior.cli")
        logger.info(f"Loaded configuration file from: {config_path}")
        logger.info(f"Configuration keys: {list(configs.keys())}")
        if "ops" in configs:
            logger.info(f"Operations to perform: {configs['ops']}")
        if "variant_effect_prediction" in configs:
            logger.info("Found variant_effect_prediction configuration")
        if "analyzer" in configs:
            logger.info("Found analyzer configuration")
        
        # Apply command line overrides to configuration
        if override:
            for o in override:
                if '=' in o:
                    param_path, value = o.split('=', 1)
                    # Implement nested dictionary traversal and value setting
                    _apply_override(configs, param_path, value)
                    logging.info(f"Applied override: {param_path}={value}")
        
        # Add debug flag to configs if specified
        if debug:
            configs["debug"] = True
        
        # Run with loaded configuration
        parse_configs_and_run(configs)
    except Exception as e:
        logging.error(f"Error running with config {config_path}: {str(e)}")
        if debug:
            import traceback
            traceback.print_exc()
        raise

@cli.command()
@click.argument('config_path', type=click.Path(exists=True, file_okay=True, dir_okay=False))
def validate(config_path: str):
    """Validate configuration file without running the model."""
    configs = load_path(config_path, instantiate=False)
    click.echo(f"Configuration valid: {config_path}")
    # Add code to print summary of configuration

def _apply_override(config: dict, param_path: str, value: str) -> None:
    """Apply override value to nested configuration."""
    # Convert value to appropriate type (int, float, bool, etc.)
    try:
        if value.lower() == 'true':
            typed_value = True
        elif value.lower() == 'false':
            typed_value = False
        elif '.' in value and value.replace('.', '').isdigit():
            typed_value = float(value)
        elif value.isdigit():
            typed_value = int(value)
        else:
            typed_value = value
    except (ValueError, AttributeError):
        typed_value = value

    # Apply to config
    keys = param_path.split('.')
    current = config
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = typed_value

if __name__ == "__main__":
    cli()  # Changed from main() to cli()

# Alias for entry point
main = cli

__all__ = ['cli', 'main']
