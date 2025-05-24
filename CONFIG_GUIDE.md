# Configuration and Model Initialization Guide for UAVarPrior

This guide explains how to ensure that configuration files are properly parsed and models are correctly initialized in UAVarPrior.

## Configuration File Structure

UAVarPrior uses YAML configuration files to define models, datasets, and operations. The basic structure of a configuration file is:

```yaml
# Operations to perform
ops:
  - train
  - evaluate

# Output directory
output_dir: ./outputs/example_run

# Model configuration
model:
  class: ModelClassName
  classArgs:
    param1: value1
    param2: value2

# Data configuration
data:
  dataset_class: path.to.DatasetClass
  batch_size: 64
  train_args:
    data_path: ./data/train.h5
  val_args:
    data_path: ./data/val.h5

# Training configuration
training:
  epochs: 10
  lr: 0.001
```

## Validating Configuration Files

UAVarPrior provides several tools to validate configurations:

### 1. Using the `validate` command

```bash
uavarprior validate /path/to/config.yml
```

This command checks:
- Required fields are present
- Operations have necessary configuration sections
- Values have the correct types
- Schema validation if jsonschema is installed

### 2. Using the `validate` command with dry-run

```bash
uavarprior validate /path/to/config.yml --dry-run
```

This command performs all basic validation and also:
- Attempts to initialize the model
- Attempts to create data loaders
- Attempts to initialize the analyzer (if specified)

### 3. Using the `debug-config` command

```bash
uavarprior debug-config /path/to/config.yml
```

This command provides detailed information about:
- Module imports
- Required functions in modules
- Path existence checks
- Detailed configuration values

You can also save the expanded configuration with default values:

```bash
uavarprior debug-config /path/to/config.yml --output expanded_config.yml
```

## Common Issues and Solutions

### Model Import Errors

If you see errors like "Could not import model module", check:
1. The model class name is correct
2. The model module is in the correct location
3. The module has the required functions: get_model, criterion, get_optimizer

### Model Initialization Errors

If model initialization fails, check:
1. All required classArgs are provided
2. The values have the correct types
3. There are no conflicts between parameters

### Loss Function Errors

If you see loss function errors, check:
1. The model module implements the criterion function
2. The criterionArgs are compatible with the loss function

### Optimizer Errors

If you see optimizer errors, check:
1. The learning rate (lr) is specified either in the model section or training section
2. The model module implements get_optimizer function

## Example Configuration Files

UAVarPrior includes example configuration files in the `config_examples` directory:
- `example_config.yml` - A complete example with train and evaluate operations
- `analyze_config.yml` - An example for running analysis operations

## Creating Custom Models

To create a custom model that works with UAVarPrior:

1. Create a Python module under uavarprior/model/nn/
2. Implement the following functions:
   - `get_model(**kwargs)` - Creates and returns your model
   - `criterion(**kwargs)` - Creates and returns your loss function
   - `get_optimizer(lr)` - Returns (optimizer_class, optimizer_kwargs)

3. Use your model in the configuration file:
```yaml
model:
  class: YourModelName
  classArgs:
    # Your model parameters
```

## Advanced Configuration

### Command-Line Overrides

You can override configuration values directly from the command line:

```bash
uavarprior run config.yml -o model.classArgs.param1=10 -o training.lr=0.01
```

### Learning Rate Override

Learning rate can be specified directly:

```bash
uavarprior run config.yml --lr 0.001
```

### Debug Mode

Enable debug mode for detailed error information:

```bash
uavarprior run config.yml --debug
```
