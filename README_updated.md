# UAVarPrior: Uncertainty-Aware Variational Prior

UAVarPrior is a framework for incorporating uncertainty-aware variational priors into machine learning models. This approach provides robust uncertainty quantification while maintaining predictive performance.

## Features

- **Variational Prior Integration**: Leverage probabilistic modeling with variational priors
- **Uncertainty Quantification**: Capture and quantify both aleatoric and epistemic uncertainty
- **Flexible Configuration**: YAML-based configuration for reproducible experiments
- **Command-line Interface**: Intuitive CLI for training, evaluation, and inference
- **Modern Model Architecture**: Clean abstraction between model interfaces and implementations

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/UAVarPrior.git
cd UAVarPrior

# Regular installation
pip install -e .

# For development (installs development dependencies)
pip install -e ".[dev]"
```

## Migration from Legacy Version

If you're migrating from an older version of UAVarPrior, run the cleanup script first:

```bash
python scripts/clean_installation.py
```

This script will clean up old installation files to avoid conflicts.

## Quick Start

### Training a Model

1. Create a configuration file (see examples in `config_examples/`)
2. Run the training command:

```bash
uavarprior run config_examples/example_config.yml
```

### Validating a Configuration

Before running a full training job, you can validate your configuration:

```bash
uavarprior validate config_examples/example_config.yml
```

For more thorough validation including model initialization:

```bash
uavarprior validate config_examples/example_config.yml --dry-run
```

### Debugging a Configuration

If you're having issues with your configuration, use the debug command:

```bash
uavarprior debug-config config_examples/example_config.yml
```

## Configuration Guide

UAVarPrior uses YAML configuration files to define models, datasets, and operations. See [CONFIG_GUIDE.md](CONFIG_GUIDE.md) for detailed information on configuration structure and options.

### Example Configuration

```yaml
# Operations to perform
ops:
  - train
  - evaluate

# Output directory
output_dir: ./outputs/example_run

# Model configuration
model:
  class: SimpleConvModel
  classArgs:
    input_channels: 4
    conv_channels: [16, 32, 64]
    kernel_size: 3
    pool_size: 2
    dropout: 0.2
    linear_features: [128, 64]
    output_features: 1

# Data configuration
data:
  dataset_class: uavarprior.data.sequences.SequenceDataset
  batch_size: 64
  train_args:
    data_path: ./data/training_data.h5
    split: train
  val_args:
    data_path: ./data/validation_data.h5
    split: val

# Training configuration
training:
  epochs: 10
  lr: 0.001
```

## Model Implementation

UAVarPrior provides a clean interface for implementing models. To create a custom model:

1. Create a PyTorch model class (nn.Module)
2. Implement the required factory functions: `get_model()`, `criterion()`, and `get_optimizer()`
3. Use your model in the configuration file

See [the example model](uavarprior/model/nn/simple_conv_model.py) for a complete implementation.

## Large Files Handling

This repository uses `.gitignore` to exclude large files that exceed GitHub's 100MB limit. Key large files are stored in designated `outputs` directories that are not tracked by Git.

### Locations for Large Files

- **Data Files**: Large data files should be stored in `uavarprior/data/outputs/`
- **Model Weights**: Large model files should be stored in `uavarprior/model/outputs/`
- **Analysis Results**: Large analysis files (like pickle files) should be stored in `uavarprior/interpret/outputs/`

## Utility Scripts

UAVarPrior includes several utility scripts to help with common tasks:

- **verify_model.py**: Verify that a model can be loaded and used
- **test_config.py**: Test a configuration file for validity

## Project Structure

The project is organized into several modules:

- **data**: For data processing and management
- **model**: For model definitions and training
- **interpret**: For result analysis and interpretation
- **analysis**: For analyzing model predictions

Each module has a `docs` directory with documentation and an `outputs` directory for large files.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the terms of the license included in the repository.
