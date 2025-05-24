"""
Utilities for loading configurations, instantiating Python objects, and
running operations in _Selene_.

"""
import os
import importlib
import sys
import re
from time import strftime
import types
import torch
import torch.nn as nn
import random
import numpy as np
import inspect

# import torchinfo  # Temporarily commented out
import torch.multiprocessing as mp

# from tensorflow.keras.models import Model

from . import instantiate
from ..model import loadNnModule
from ..model import loadWrapperModule
from ..model.nn.utils import load_model, add_output_layers, make_dir

from torch.distributed import init_process_group, destroy_process_group
from datetime import timedelta
import logging
from typing import Dict, Any
import torch

from uavarprior.model import get_model
# Commented out missing imports to fix compatibility issues
# Original import: from uavarprior.data import get_dataset, get_dataloader
# Fixed import: changed training to train and Trainer to StandardSGDTrainer
from uavarprior.train import StandardSGDTrainer

from .helper import try_import_peakgvarevaluator

logger = logging.getLogger(__name__)

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank,
                       world_size=world_size,
                       timeout=timedelta(minutes=30))
    torch.cuda.set_device(rank)

def class_instantiate(classobj):
    """Not used currently, but might be useful later for recursive
    class instantiation
    """
    for attr, obj in classobj.__dict__.items():
        is_module = getattr(obj, '__module__', None)
        if is_module and "uavarprior" in is_module and attr != "model":
            class_instantiate(obj)
    classobj.__init__(**classobj.__dict__)


def module_from_file(path):
    """
    Load a module created based on a Python file path.

    Parameters
    ----------
    path : str
        Path to the model architecture file.

    Returns
    -------
    The loaded module

    """
    parent_path, module_file = os.path.split(path)
    loader = importlib.machinery.SourceFileLoader(
        module_file[:-3], path)
    module = types.ModuleType(loader.name)
    loader.exec_module(module)
    return module


def module_from_dir(path):
    """
    This method expects that you pass in the path to a valid Python module,
    where the `__init__.py` file already imports the model class,
    `criterion`, and `get_optimizer` methods from the appropriate file
    (e.g. `__init__.py` contains the line `from <model_class_file> import
    <ModelClass>`).

    Parameters
    ----------
    path : str
        Path to the Python module containing the model class.

    Returns
    -------
    The loaded module
    """
    parent_path, module_dir = os.path.split(path)
    sys.path.insert(0, parent_path)
    return importlib.import_module(module_dir)

def _getModelInfo(configs, sampler):
    '''
    Assemble model info from the config dictionary
    '''
    modelInfo = configs["model"]
    if not ('classArgs' in modelInfo):
        modelInfo['classArgs'] = dict()
    classArgs = modelInfo['classArgs']
    if not ('sequence_length' in classArgs):
        classArgs['sequence_length'] = sampler.getSequenceLength()
    if not ('n_targets' in classArgs):
        classArgs['n_targets'] = len(sampler.getFeatures())
    
    return modelInfo

def initialize_model(model_configs, train=True, lr=None, configs=None):
    """
    Initialize model (and associated criterion, optimizer)

    Parameters
    ----------
    model_configs : dict
        Model-specific configuration
    train : bool, optional
        Default is True. If `train`, returns the user-specified optimizer
        and optimizer class that can be found within the input model file.
    lr : float or None, optional
        If `train`, a learning rate must be specified. Otherwise, None.

    Returns
    -------
    model, criterion : tuple(torch.nn.Module, torch.nn._Loss) or \
            model, criterion, optim_class, optim_kwargs : \
                tuple(torch.nn.Module, torch.nn._Loss, torch.optim, dict)

        * `torch.nn.Module` - the model architecture
        * `torch.nn._Loss` - the loss function associated with the model
        * `torch.optim` - the optimizer associated with the model
        * `dict` - the optimizer arguments

        The optimizer and its arguments are only returned if `train` is
        True.

    Raises
    ------
    ValueError
        If `train` but the `lr` specified is not a float.

    """
    if model_configs["built"] == 'pytorch':
        model_class_name = model_configs["class"]

        if 'path' in model_configs.keys():
            # load network module from user given file
            import_model_from = model_configs["path"]
            if os.path.isdir(import_model_from):
                module = module_from_dir(import_model_from)
            else:
                module = module_from_file(import_model_from)
        else:
            # load built in network module
            module = loadNnModule(model_class_name)

        model_class = getattr(module, model_class_name)
        ### Get only exprected arguments and ignore extra args
        model_class_expected_argset = set(inspect.getargspec(model_class).args)
        model_class_args = {k: model_configs["classArgs"][k] for k in model_class_expected_argset if k in model_configs["classArgs"]}

        # model = model_class(**model_configs["classArgs"])
        model = model_class(**model_class_args)

        def xavier_initialize(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if 'xavier_init' in model_configs:
            model.apply(xavier_initialize)


        if "non_strand_specific" in model_configs:
            from uavarprior.model import NonStrandSpecific
            model = NonStrandSpecific(
                model, mode=model_configs["non_strand_specific"])

        # loss function
        if 'criterionArgs' in model_configs:
            criterionArgs = model_configs['criterionArgs']
        else:
            criterionArgs = dict()
        criterion = module.criterion(**criterionArgs)

        # optimizer for training
        optim_class, optim_kwargs = None, None
        if train:
            if isinstance(lr, float):
                optim_class, optim_kwargs = module.get_optimizer(lr)
            elif lr is not None:
                # Try to convert lr to float if possible
                try:
                    lr_float = float(lr)
                    logger.info(f"Converting learning rate from {type(lr).__name__} ({lr}) to float ({lr_float})")
                    optim_class, optim_kwargs = module.get_optimizer(lr_float)
                except (ValueError, TypeError):
                    logger.error(f"Failed to convert learning rate value: {lr} (type: {type(lr).__name__})")
                    raise ValueError("Learning rate must be convertible to a float "
                                     f"but was {lr} of type {type(lr).__name__}")
            else:
                # If we're in training mode but no learning rate provided, this is an error
                raise ValueError("Learning rate must be specified for training mode")
        # elif train:
        #     # This branch should never be reached with our new changes above
        #     optim_class, optim_kwargs = module.get_optimizer(lr)
    # elif model_configs["built"] == 'tensorflow':
    #     model_class_name = model_configs["class"]
    #     # model_built_name = model_configs["built"]
    #
    #     module = None
    #     if 'path' in model_configs.keys():
    #         # load network module from user given file
    #         import_model_from = model_configs["path"]
    #         if os.path.isdir(import_model_from):
    #             module = module_from_dir(import_model_from)
    #         else:
    #             module = module_from_file(import_model_from)
    #     else:
    #         # load built in network module
    #         module = loadNnModule(model_class_name)
    #
    #     model_class = getattr(module, model_class_name)
    #
    #     model_class_expected_argset = set(inspect.getargspec(model_class).args)
    #     # model_class_args = {k: model_configs["classArgs"][k] for k in model_class_expected_argset if
    #     #                     k in model_configs["classArgs"]}
    #
    #
    #     model_builder = model_class(**model_configs["classArgs"])
    #     # model_builder = model_class(**model_class_args)
    #     dna_wlen = model_configs["dna_wlen"]
    #     dna_inputs = model_builder.inputs(dna_wlen)
    #     stem = model_builder(dna_inputs)
    #     output_names = model_configs['output_names']
    #
    #     # loss function
    #     if 'criterionArgs' in model_configs:
    #         criterionArgs = model_configs['criterionArgs']
    #     else:
    #         criterionArgs = dict()
    #     criterion = module.criterion(**criterionArgs)
    #
    #     outputs = add_output_layers(stem.outputs[0], output_names,
    #                                 loss_fn=criterion)
    #     # from tensorflow.keras.models import Model
    #     model = Model(inputs=stem.inputs, outputs=outputs, name=stem.name)
    #
    #     if "non_strand_specific" in model_configs:
    #         from uavarprior.model import NonStrandSpecific
    #         model = NonStrandSpecific(
    #             model, mode=model_configs["non_strand_specific"])
    #
    #
    #
    #     # optimizer for training
    #     optim_class, optim_kwargs = None, None
    #     if train and isinstance(lr, float):
    #         optim_class, optim_kwargs = module.get_optimizer(lr)
    #     elif train:
    #         raise ValueError("Learning rate must be specified as a float "
    #                          "but was {0}".format(lr))

        # if 'path' in model_configs.keys():
        #     model = load_model(model_configs["path"])
        #
        # model_class_name = model_configs["class"]
        # module = loadNnModule(model_class_name)
        # # TO DO: raise error when non strand specific in configuration variables
        # # if "non_strand_specific" in model_configs:
        # #     from uavarprior.model import NonStrandSpecific
        # #     model = NonStrandSpecific(
        # #         model, mode=model_configs["non_strand_specific"])
        #
        # # loss function
        # if 'criterionArgs' in model_configs:
        #     criterionArgs = model_configs['criterionArgs']
        # else:
        #     criterionArgs = dict()
        # criterion = module.criterion(**criterionArgs)
        #
        # # optimizer for training
        # optim_class, optim_kwargs = None, None
        # if train and isinstance(lr, float):
        #     optim_class, optim_kwargs = module.get_optimizer(lr)
        # elif train:
        #     raise ValueError("Learning rate must be specified as a float "
        #                      "but was {0}".format(lr))

    # construct model wrapper
    if 'plot_grads' in model_configs:
        plot_grads = model_configs['plot_grads']
        if plot_grads:
            gradOutDir = configs['output_dir']
            make_dir(gradOutDir)
    else:
        gradOutDir = None
    if 'rank' in model_configs.keys():
        rank = model_configs['rank']
    else:
        rank = None

    modelWrapper = initializeWrapper(model_configs['wrapper'], 
         mode = 'train', model = model, loss = criterion,
            model_built = model_configs['built'], mult_predictions = model_configs['mult_predictions'],
         optimizerClass = optim_class,  optimizerKwargs = optim_kwargs,
            gradOutDir=gradOutDir, rank=rank
                                     )
    # modelWrapper._model.built = model_configs["built"]
    return modelWrapper

def initializeWrapper(className, mode, model, loss, model_built = 'pytorch', mult_predictions=1, useCuda = None,
          optimizerClass = None, optimizerKwargs = None,
                      gradOutDir=None, rank=None
                      ):
    '''
    Initialize model wrapper
    '''
    wrapperClass = getattr(loadWrapperModule(className), className)
    wrapper = wrapperClass(model, mode = mode, lossCalculator = loss,
                           model_built = model_built, mult_predictions=mult_predictions,
             useCuda = useCuda, optimizerClass = optimizerClass, 
             optimizerKwargs = optimizerKwargs,
                gradOutDir=gradOutDir, rank=rank
                )
    return wrapper
    
# Helper functions for class importing to be used in execute()
def _import_class(class_path):
    """Helper to import a class from its string path"""
    if class_path is None:
        logger.debug("Class path is None, cannot import")
        return None
        
    if '.' in class_path:
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            logger.debug(f"Importing module: {module_path}, class: {class_name}")

            # Define possible import paths based on module_path
            base_path = module_path.replace('src.', '').replace('uavarprior.', '')
            possible_paths = [
                f"src.uavarprior.{base_path}",  # Most specific path first
                f"uavarprior.{base_path}",
                f"src.uavarprior.{base_path.rsplit('.', 1)[0]}",  # Try parent package
                f"uavarprior.{base_path.rsplit('.', 1)[0]}",
                "src.uavarprior",  # Legacy paths as fallback
                "uavarprior"
            ]

            # Try each import path in order
            for try_path in possible_paths:
                try:
                    logger.debug(f"Attempting import from {try_path}")
                    module = importlib.import_module(try_path)
                    if hasattr(module, class_name):
                        return getattr(module, class_name)
                except (ImportError, AttributeError) as e:
                    logger.debug(f"Import failed for {try_path}: {e}")
                    continue
                    
            logger.debug(f"All import attempts failed for {class_path}")
            return None
            
        except Exception as e:
            logger.debug(f"Error parsing class path {class_path}: {e}")
            return None
    else:
        # If no module path is specified, assume it's a class in the current module
        logger.debug(f"No module specified, treating {class_path} as a simple class name")
        # This is handled by _import_from_module, so return None
        return None
        
def _import_from_module(module_path, class_name):
    """Helper to import a specific class from a module"""
    if module_path is None or class_name is None:
        logger.debug("Module path or class name is None, cannot import")
        return None
        
    logger.debug(f"Importing from module: {module_path}, class: {class_name}")
    
    # Define possible import paths based on module_path
    base_path = module_path.replace('src.', '').replace('uavarprior.', '')
    possible_paths = [
        f"src.uavarprior.{base_path}",  # Most specific path first
        f"uavarprior.{base_path}",
        f"src.uavarprior.{base_path.rsplit('.', 1)[0] if '.' in base_path else base_path}",  # Try parent package
        f"uavarprior.{base_path.rsplit('.', 1)[0] if '.' in base_path else base_path}",
        "src.uavarprior",  # Legacy paths as fallback
        "uavarprior"
    ]
    
    # Try each import path in order
    for try_path in possible_paths:
        try:
            logger.debug(f"Attempting import from {try_path}")
            module = importlib.import_module(try_path)
            if hasattr(module, class_name):
                return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            logger.debug(f"Import failed for {try_path}: {e}")
            continue
            
    logger.debug(f"All import attempts failed for {class_name} from {module_path}")
    return None

def execute(configs):
    """
    Execute operations in _Selene_.

    Parameters
    ----------
    # operations : list(str)
    #     The list of operations to carry out in _Selene_.
    configs : dict or object
        The loaded configurations from a YAML file.
    output_dir : str or None
        The path to the directory where all outputs will be saved.
        If None, this means that an `output_dir` was not specified
        in the top-level configuration keys. `output_dir` must be
        specified in each class's individual configuration wherever
        it is required.

    Returns
    -------
    None
        Executes the operations listed and outputs any files
        to the dirs specified in each operation's configuration.

    Raises
    ------
    ValueError
        If an expected key in configuration is missing.

    """

    model = None
    modelTrainer = None
    output_dir = configs['output_dir']
    for op in configs['ops']:
        if op == "train":
            # ddp_setup(rank, world_size=torch.cuda.device_count())
            # construct sampler
            sampler_info = configs["sampler"]
            if output_dir is not None:
                sampler_info.bind(output_dir=output_dir)
            sampler = instantiate(sampler_info)

            # construct model
            modelInfo = _getModelInfo(configs, sampler)

            # modelInfo['rank'] = rank
            model = initialize_model(modelInfo, train = True,
                                     lr = configs["lr"], configs=configs)

            # ## Create Distributed strategy
            # ddp_setup(rank, world_size=torch.cuda.device_count())
            
            # create trainer
            train_model_info = configs["train_model"]
            train_model_info.bind(model = model, dataSampler = sampler)
            if sampler.getValOfMisInTarget() is not None:
                train_model_info.bind(valOfMisInTarget = sampler.getValOfMisInTarget())
            
            if output_dir is not None:
                train_model_info.bind(outputDir = output_dir)
            if "random_seed" in configs:
                train_model_info.bind(deterministic=True)

            modelTrainer = instantiate(train_model_info)
            
            # train
            modelTrainer.trainAndValidate()
            # destroy_process_group()

        elif op == "evaluate":
            if modelTrainer is not None:
                modelTrainer.evaluate()
            else:
                # construct sampler
                sampler_info = configs["sampler"]
                sampler = instantiate(sampler_info)
                
                # construct model
                modelInfo = _getModelInfo(configs, sampler)
                model = initialize_model(modelInfo, train = False)
                
                # construct evaluator  
                evaluate_model_info = configs["evaluate_model"]
                evaluate_model_info.bind(model = model, dataSampler = sampler)
                if output_dir is not None:
                    evaluate_model_info.bind(outputDir = output_dir)
                if sampler.getValOfMisInTarget() is not None:
                    evaluate_model_info.bind(valOfMisInTarget = sampler.getValOfMisInTarget())
                evaluate_model = instantiate(evaluate_model_info)
                
                # evaluate
                evaluate_model.evaluate()

        elif op == "analyze":
            logger.info("Processing analyze operation")
            if not model:
                logger.info("No model already loaded, initializing model from config")
                try:
                    # lr=None because we don't need optimizer for analysis
                    model_config = configs["model"]
                    # Add debugging information
                    logger.debug(f"Model config before initialization: {model_config}")
                    
                    # Check if model_config is a dictionary and has required fields
                    if isinstance(model_config, dict) and "built" in model_config and "wrapper" in model_config:
                        model = initialize_model(model_config, train=False, lr=None, configs=configs)
                        logger.info("Model initialized successfully")
                    else:
                        # More detailed error message for debugging
                        logger.error(f"Invalid model configuration: {model_config}")
                        raise ValueError(f"Model configuration must be a dictionary with 'built' and 'wrapper' keys")
                except Exception as e:
                    logger.error(f"Failed to initialize model: {str(e)}")
                    raise
            
            # construct analyzer
            logger.info("Setting up analyzer")
            analyze_seqs_info = configs["analyzer"]
            
            # Check if analyze_seqs_info is a proxy object or a dict
            if hasattr(analyze_seqs_info, 'bind'):
                # It's a _Proxy object with bind method
                analyze_seqs_info.bind(model=model)
                if output_dir is not None:
                    analyze_seqs_info.bind(outputDir=output_dir)
                analyze_seqs = instantiate(analyze_seqs_info)
            else:
                # It's a plain dict, we need to add model and outputDir to the dict
                logger.info("Analyzer config is a dictionary, adding parameters directly")
                if isinstance(analyze_seqs_info, dict):
                    analyze_seqs_info['model'] = model
                    if output_dir is not None:
                        analyze_seqs_info['outputDir'] = output_dir
                    # Now instantiate the class directly
                    try:
                        # Store a copy of the original class path before popping it
                        original_class_path = analyze_seqs_info.get('class')
                        logger.info(f"Original class path from config: {original_class_path}")
                        
                        # CRITICAL FIX: Keep a copy of the original analyzer config before modifying
                        original_analyze_seqs_info = analyze_seqs_info.copy()
                        
                        # Use the class name from the dict directly
                        class_path = analyze_seqs_info.pop('class', None)
                        
                        # Check for '!obj:' tag pattern in the original analyzer config or in the configs
                        if not class_path:
                            # First try to get from the original analyzer config
                            yaml_repr = str(original_analyze_seqs_info)
                            logger.debug(f"Checking for !obj: tag in: {yaml_repr}")
                            if '!obj:' in yaml_repr:
                                # Try to extract from YAML tag if available
                                match = re.search(r'!obj:([\w\.]+)', yaml_repr)
                                if match:
                                    class_path = match.group(1)
                                    logger.info(f"Extracted class path from analyzer !obj: tag: {class_path}")
                                else:
                                    # If we know it's a YAML tag but couldn't extract, log more details
                                    logger.warning(f"Found !obj: tag but couldn't extract class path. Raw config: {yaml_repr}")
                            
                            # If still no class_path, check the full configs
                            if not class_path and isinstance(configs["analyzer"], str):
                                raw_analyzer = configs["analyzer"]
                                logger.debug(f"Checking raw analyzer string: {raw_analyzer}")
                                match = re.search(r'!obj:([\w\.]+)', raw_analyzer)
                                if match:
                                    class_path = match.group(1)
                                    logger.info(f"Extracted class path from raw config !obj: tag: {class_path}")
                        
                        # For PeakGVarEvaluator, check if it's a typical class name without module prefix
                        if class_path and class_path.endswith("PeakGVarEvaluator") and "." not in class_path:
                            logger.info(f"Detected bare PeakGVarEvaluator class name, assuming src.uavarprior.predict.seq_ana.gve.PeakGVarEvaluator")
                            class_path = f"src.uavarprior.predict.seq_ana.gve.{class_path}"
                        # If it's uavarprior.predict.PeakGVarEvaluator, convert to the correct path 
                        elif class_path and "uavarprior.predict.PeakGVarEvaluator" in class_path and "seq_ana" not in class_path:
                            new_class_path = class_path.replace("uavarprior.predict.PeakGVarEvaluator", "uavarprior.predict.seq_ana.gve.PeakGVarEvaluator")
                            logger.info(f"Fixed class path from {class_path} to {new_class_path}")
                            class_path = new_class_path
                        
                        logger.info(f"Final extracted class path: {class_path}")
                        
                        if class_path:
                            logger.info(f"Instantiating analyzer from class path: {class_path}")
                            # Store original for error reporting
                            original_analyzer_config = {
                                'class': original_class_path,
                                **analyze_seqs_info
                            }
                            class_obj = None
                            exceptions = []
                            
                            # Try multiple import strategies in order
                            import_strategies = [
                                # Strategy 1: Import from GVE module directly (most specific path)
                                lambda: _import_from_module("src.uavarprior.predict.seq_ana.gve", "PeakGVarEvaluator"),
                                lambda: _import_from_module("uavarprior.predict.seq_ana.gve", "PeakGVarEvaluator"),
                                
                                # Strategy 2: Import from seq_ana package
                                lambda: _import_from_module("src.uavarprior.predict.seq_ana", "PeakGVarEvaluator"),
                                lambda: _import_from_module("uavarprior.predict.seq_ana", "PeakGVarEvaluator"),
                                
                                # Strategy 3: Legacy - try predict module directly
                                lambda: _import_from_module("src.uavarprior.predict", "PeakGVarEvaluator"),
                                lambda: _import_from_module("uavarprior.predict", "PeakGVarEvaluator"),
                            ]

                            # Try each strategy until one works
                            success = False
                            last_error = None
                            for strategy in import_strategies:
                                try:
                                    logger.debug(f"Trying import strategy: {strategy.__name__ if hasattr(strategy, '__name__') else 'anonymous'}")
                                    analyzer_cls = strategy()
                                    if analyzer_cls is not None:
                                        analyze_seqs = analyzer_cls(**analyze_seqs_info)
                                        success = True
                                        logger.info(f"Successfully imported PeakGVarEvaluator using strategy {strategy.__name__ if hasattr(strategy, '__name__') else 'anonymous'}")
                                        break
                                except Exception as e:
                                    last_error = e
                                    logger.debug(f"Import strategy failed: {e}")
                                    continue

                            if not success:
                                error_msg = f"All import strategies failed. Last error: {last_error}"
                                logger.error(error_msg)
                                raise ImportError(error_msg)
                        else:
                            logger.error("No class path found in configuration")
                            logger.error(f"Original analyzer configuration: {original_analyze_seqs_info}")
                            
                            # Try a hard-coded default as a last-ditch effort
                            default_class_path = "src.uavarprior.predict.seq_ana.gve.PeakGVarEvaluator"
                            logger.info(f"Attempting to use default analyzer: {default_class_path}")
                            
                            try:
                                # Try multiple import strategies for the default class path in specified order
                                successful_import = False
                                import_paths = [
                                    "src.uavarprior.predict.seq_ana.gve",  # Most specific path first
                                    "uavarprior.predict.seq_ana.gve",  
                                    "src.uavarprior.predict.seq_ana",  # Try seq_ana package
                                    "uavarprior.predict.seq_ana",
                                    "src.uavarprior.predict",  # Legacy paths as fallback 
                                    "uavarprior.predict"
                                ]
                                
                                for import_path in import_paths:
                                    try:
                                        logger.debug(f"Trying to import PeakGVarEvaluator from {import_path}")
                                        module = importlib.import_module(import_path)
                                        if hasattr(module, "PeakGVarEvaluator"):
                                            logger.info(f"Found PeakGVarEvaluator in {import_path}")
                                            analyze_seqs = module.PeakGVarEvaluator(**analyze_seqs_info)
                                            successful_import = True
                                            break
                                    except ImportError as e:
                                        logger.debug(f"Could not import module {import_path}: {e}")
                                        # Continue to next path
                                        continue
                                
                                if not successful_import:
                                    logger.info("All module imports failed, trying with individual class paths")
                                    class_paths = [
                                        "src.uavarprior.predict.seq_ana.gve.PeakGVarEvaluator",
                                        "uavarprior.predict.seq_ana.gve.PeakGVarEvaluator"
                                    ]
                                    for path in class_paths:
                                        try:
                                            logger.debug(f"Attempting direct class import from {path}")
                                            cls = _import_class(path)
                                            if cls:
                                                logger.info(f"Successfully imported PeakGVarEvaluator from {path}")
                                                analyze_seqs = cls(**analyze_seqs_info)
                                                successful_import = True
                                                break
                                        except Exception as e:
                                            logger.debug(f"Direct class import failed from {path}: {e}")
                                
                                if not successful_import:
                                    raise ImportError("Could not import PeakGVarEvaluator from any known location")
                            except Exception as e:
                                logger.error(f"Failed to use default analyzer: {e}")
                                logger.error("Please update your configuration to include a valid 'class' property in the 'analyzer' section")
                                raise ValueError("Could not determine analyzer class from configuration. Make sure 'class' is specified in the analyzer configuration.")
                    except Exception as e:
                        logger.error(f"Failed to instantiate analyzer: {str(e)}")
                        raise
                else:
                    raise TypeError(f"Expected analyzer config to be a dict or proxy object, got {type(analyze_seqs_info).__name__}")
            
            logger.info("Analyzer set up complete, determining analysis type")
            if "variant_effect_prediction" in configs:
                logger.info("Running variant effect prediction analysis")
                analyze_seqs.evaluate()
            elif "in_silico_mutagenesis" in configs:
                logger.info("Running in silico mutagenesis")
                ism_info = configs["in_silico_mutagenesis"]
                analyze_seqs.evaluate(**ism_info)
            elif "prediction" in configs:
                logger.info("Running prediction")
                predict_info = configs["prediction"]
                analyze_seqs.predict(**predict_info)
            else:
                raise ValueError('The type of analysis needs to be specified. It can '
                               'either be variant_effect_prediction, in_silico_mutagenesis '
                               'or prediction')


def parse_configs_and_run(configs: Dict[str, Any]) -> None:
    """Parse configuration and run the specified task.
    
    Args:
        configs: Dictionary containing configuration parameters
    """
    try:
        # Add import re if needed
        import re
        
        # Enhanced detection of configuration formats
        # Check if this is a legacy FuGEP-style config with ops key
        if "ops" in configs:
            ops = configs.get("ops", [])
            logger.info(f"Detected legacy configuration format with operations: {ops}")
            return execute(configs)
            
        # Check if it's an analyze-only configuration without ops
        if ("analyzer" in configs and ("variant_effect_prediction" in configs or 
                                     "in_silico_mutagenesis" in configs or
                                     "prediction" in configs)):
            logger.info("Detected analyze-only configuration without 'ops' key. Adding ops=['analyze'].")
            temp_configs = configs.copy()
            temp_configs["ops"] = ["analyze"]
            return execute(temp_configs)
            
        # Check for CUDA availability and set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Extract key configurations
        data_config = configs.get("data", {})
        model_config = configs.get("model", {})
        training_config = configs.get("training", {})
        metrics = training_config.get("metrics", [])
        # Output directory
        output_dir = configs.get("output_dir", None)

        # Initialize sampler as data source and loader
        logger.info("Initializing sampler")
        sampler_info = configs.get("sampler")
        
        # Debug logging to see what's in the configs
        logger.debug(f"Available config keys: {list(configs.keys())}")
        if not sampler_info:
            # If it's not a clear legacy config but has analyzer section, it's likely an analyze-only config
            if "analyzer" in configs and ("variant_effect_prediction" in configs or 
                                         "in_silico_mutagenesis" in configs or
                                         "prediction" in configs):
                logger.info("Detected analyze-only configuration. Using execute() function.")
                temp_configs = configs.copy()
                temp_configs["ops"] = ["analyze"]  # Add ops to make execute work
                return execute(temp_configs)
                
            # If we get here, the config doesn't have a valid structure
            raise ValueError("Invalid configuration: Missing 'sampler' key and not a recognized legacy format. Check your YAML structure.")
        
        sampler = instantiate(sampler_info)
        
        # Get learning rate and ensure it's a float
        lr_value = training_config.get('lr')
        if lr_value is not None:
            try:
                lr_value = float(lr_value)
            except (ValueError, TypeError):
                raise ValueError(f"Learning rate must be a valid number, but got: {lr_value}")
        else:
            # If learning rate from training_config is None, try to get it from the top-level configs
            lr_value = configs.get('lr')
            if lr_value is not None:
                try:
                    lr_value = float(lr_value)
                except (ValueError, TypeError):
                    raise ValueError(f"Learning rate must be a valid number, but got: {lr_value}")
            else:
                logger.warning("No learning rate specified in config. This may cause issues if training is enabled.")
                
        # initialize_model returns a wrapper with getOptimizer, setOptimizer, etc.
        model = initialize_model(
            model_config,
            train=True,
            lr=lr_value,
            configs=configs
        )

        # Setup training
        logger.info("Setting up training...")
        trainer = StandardSGDTrainer(
            model=model,
            dataSampler=sampler,
            outputDir=configs.get("output_dir"),
            maxNSteps=training_config.get("max_steps", None),
            batchSize=training_config.get("batch_size", 64),
            useCuda=(device == "cuda"),
            dataParallel=training_config.get("data_parallel", False),
            loggingVerbosity=training_config.get("logging_verbosity", 2),
            metrics=metrics,
        )
        
        # Run training or inference based on mode
        mode = configs.get("mode", "train")
        if mode == "train":
            logger.info("Starting training...")
            trainer.trainAndValidate()
        elif mode == "evaluate":
            logger.info("Starting evaluation...")
            trainer.evaluate()
        elif mode == "predict":
            logger.info("Starting prediction...")
            trainer.predict()
        else:
            raise ValueError(f"Unknown execution mode: {mode}")
    except Exception as e:
        logger.error(f"Execution failed: {str(e)}")
        if configs.get("debug", False):
            import traceback
            traceback.print_exc()
        raise
