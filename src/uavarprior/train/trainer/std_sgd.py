"""
This module provides the `TrainModel` class and supporting methods.
"""
import logging

import math
from time import time
import os
import numpy as np
import torch
import torch.nn as nn


from ..utils import LossTracker, load_model_from_state_dict

from .sgd_trainer import SGDTrainer


logger = logging.getLogger("fugep")
# import torchinfo  # Temporarily commented out

def timeit(func):
    import time
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        print(f'Time taken for {func}: {elapsed:.6f} seconds')
        return result
    return wrapper




class StandardSGDTrainer(SGDTrainer):
    """
    This class ties together the various objects and methods needed to
    train and validate a model.

    TrainModel saves a checkpoint model (overwriting it after
    `save_checkpoint_every_n_steps`) as well as a best-performing model
    (overwriting it after `report_stats_every_n_steps` if the latest
    validation performance is better than the previous best-performing
    model) to `output_dir`.

    TrainModel also outputs 2 files that can be used to monitor training
    as Selene runs: `fugep.train_model.train.txt` (training loss) and
    `fugep.train_model.validation.txt` (validation loss & average
    ROC AUC). The columns in these files can be used to quickly visualize
    training history (e.g. you can use `matplotlib`, `plt.plot(auc_list)`)
    and see, for example, whether the model is still improving, if there are
    signs of overfitting, etc.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train.
    data_sampler : fugep.samplers.Sampler
        The example generator.
    loss_criterion : torch.nn._Loss
        The loss function to optimize.
    optimizer_class : torch.optim.Optimizer
        The optimizer to minimize loss with.
    optimizer_kwargs : dict
        The dictionary of keyword arguments to pass to the optimizer's
        constructor.
    batch_size : int
        Specify the batch size to process examples. Should be a power of 2.
    max_steps : int
        The maximum number of mini-batches to iterate over.
    report_stats_every_n_steps : int
        The frequency with which to report summary statistics. You can
        set this value to be equivalent to a training epoch
        (`n_steps * batch_size`) being the total number of samples
        seen by the model so far. Selene evaluates the model on the validation
        dataset every `report_stats_every_n_steps` and, if the model obtains
        the best performance so far (based on the user-specified loss function),
        Selene saves the model state to a file called `best_model.pth.tar` in
        `output_dir`.
    output_dir : str
        The output directory to save model checkpoints and logs in.
    save_checkpoint_every_n_steps : int or None, optional
        Default is 1000. If None, set to the same value as
        `report_stats_every_n_steps`
    save_new_checkpoints_after_n_steps : int or None, optional
        Default is None. The number of steps after which Selene will
        continually save new checkpoint model weights files
        (`checkpoint-<TIMESTAMP>.pth.tar`) every
        `save_checkpoint_every_n_steps`. Before this point,
        the file `checkpoint.pth.tar` is overwritten every
        `save_checkpoint_every_n_steps` to limit the memory requirements.
    n_validation_samples : int or None, optional
        Default is `None`. Specify the number of validation samples in the
        validation set. If `n_validation_samples` is `None` and the data sampler
        used is the `fugep.samplers.IntervalsSampler` or
        `fugep.samplers.RandomSampler`, we will retrieve 32000
        validation samples. If `None` and using
        `fugep.samplers.MultiSampler`, we will use all
        available validation samples from the appropriate data file.
    n_test_samples : int or None, optional
        Default is `None`. Specify the number of test samples in the test set.
        If `n_test_samples` is `None` and

            - the sampler you specified has no test partition, you should not
              specify `evaluate` as one of the operations in the `ops` list.
              That is, Selene will not automatically evaluate your trained
              model on a test dataset, because the sampler you are using does
              not have any test data.
            - the sampler you use is of type `fugep.samplers.OnlineSampler`
              (and the test partition exists), we will retrieve 640000 test
              samples.
            - the sampler you use is of type
              `fugep.samplers.MultiSampler` (and the test partition
              exists), we will use all the test samples available in the
              appropriate data file.

    cpu_n_threads : int, optional
        Default is 1. Sets the number of OpenMP threads used for parallelizing
        CPU operations.
    use_cuda : bool, optional
        Default is `False`. Specify whether a CUDA-enabled GPU is available
        for torch to use during training.
    data_parallel : bool, optional
        Default is `False`. Specify whether multiple GPUs are available
        for torch to use during training.
    logging_verbosity : {0, 1, 2}, optional
        Default is 2. Set the logging verbosity level.

            * 0 - Only warnings will be logged.
            * 1 - Information and warnings will be logged.
            * 2 - Debug messages, information, and warnings will all be\
                  logged.

    checkpointResume : str or None, optional
        Default is `None`. If `checkpointResume` is not None, it should be the
        path to a model file generated by `torch.save` that can now be read
        using `torch.load`.
    use_scheduler : bool, optional
        Default is `True`. If `True`, learning rate scheduler is used to
        reduce learning rate on plateau. PyTorch ReduceLROnPlateau scheduler 
        with patience=16 and factor=0.8 is used.

    Attributes
    ----------
    model : torch.nn.Module
        The model to train.
    sampler : fugep.samplers.Sampler
        The example generator.
    criterion : torch.nn._Loss
        The loss function to optimize.
    optimizer : torch.optim.Optimizer
        The optimizer to minimize loss with.
    batch_size : int
        The size of the mini-batch to use during training.
    max_steps : int
        The maximum number of mini-batches to iterate over.
    nth_step_report_stats : int
        The frequency with which to report summary statistics.
    nth_step_save_checkpoint : int
        The frequency with which to save a model checkpoint.
    use_cuda : bool
        If `True`, use a CUDA-enabled GPU. If `False`, use the CPU.
    data_parallel : bool
        Whether to use multiple GPUs or not.
    output_dir : str
        The directory to save model checkpoints and logs.

    """

    def __init__(self,
                 model,
                 dataSampler,
                 outputDir,
                 maxNSteps, # TODO: default None, use early stopping
                 lossCalculator = None,
                 optimizerClass = None,
                 optimizerKwargs = None,
                 batchSize = 64,
                 nStepsStatReport = 100,
                 nStepsCheckpoint = 1000,
                 nStepsStartCheckpoint = None,
                 nMinMinorsReport = 10,
                 nValidationSamples=None,
                 nTestSamples=None,
                 nCpuThreads = 1,
                 useCuda = False,
                 dataParallel = False,
                 loggingVerbosity=2,
                 preloadValData=False,
                 preloadTestData=False,
                 gradOutDir=None,
                 schedulerName=None,
                 changeOptim=False,
                 checkpointResume = None,
                 transferAndTune = None,
                 newClassifier = None,
                 freezeStem = None,
                 metrics = None,
                 useScheduler=True,
                 deterministic=False,
                 valOfMisInTarget = None):
        """
        Constructs a new `StandardSGDTrainer` object.
        """
        super(StandardSGDTrainer, self).__init__(
            model = model,
            dataSampler = dataSampler,
            lossCalculator = lossCalculator,
            optimizerClass = optimizerClass,
            optimizerKwargs = optimizerKwargs,
            outputDir = outputDir,
            maxNSteps = maxNSteps, 
            batchSize = batchSize,
            nStepsStatReport = nStepsStatReport,
            nStepsCheckpoint = nStepsCheckpoint,
            nStepsStartCheckpoint = nStepsStartCheckpoint,
            nMinMinorsReport = nMinMinorsReport,
            nValidationSamples = nValidationSamples,
            nTestSamples = nTestSamples,
            nCpuThreads = nCpuThreads,
            useCuda = useCuda,
            dataParallel = dataParallel,
            loggingVerbosity = loggingVerbosity,
            preloadValData = preloadValData,
            preloadTestData = preloadTestData,
            schedulerName = schedulerName,
            changeOptim = changeOptim,
            gradOutDir = gradOutDir,
            metrics = metrics,
            useScheduler = useScheduler,
            deterministic = deterministic,
            valOfMisInTarget = valOfMisInTarget)
        
        if checkpointResume is not None:
            self._loadCheckpoint(checkpointResume)
        if transferAndTune:
            self._loadTrainedModel(transferAndTune, newClassifier, freezeStem)


    # @profile
    # @timeit
    def trainAndValidate(self):
        """
        Trains the model and measures validation performance.

        """

        # if self.model.rank == 0:
        # logger.info("Pytorch model summary: {}".format(torchinfo.summary(
        #                         self.model._model,
        #                         input_size=(self.batchSize, 4,
        #                                     self.sampler.getSequenceLength())
        #                     )))  # Temporarily commented out

        validationPerformance = os.path.join(self.outputDir, "val_metrics.txt")

        for step in range(self._startStep, self.maxNSteps):
            self.step = step
            self._train()

            if step % self.nStepsCheckpoint == 0:
                # logger.info("Device rank: {}".format(self.model.rank))
                if self.model.rank == 0:
                    self._checkpoint()
            if self.step and self.step % self.nStepsStatReport == 0:
                self._validate(step)
                # logger.info("Device {0} Learning rate: {1}".format(self.model.rank, self.scheduler._last_lr))
                # logger.info("Learning Rate: {}".format(self.scheduler.get_last_lr()[0]))
        self._validationMetrics.writeValidationFeatureScores(
            validationPerformance)

        # self.sampler.saveDatasetToFile("train", close_filehandle = True)
        logger.info("Training and validation are complete and metrics saved")


    # @profile
    # @timeit
    def _train(self):
        """
        Trains the model on a batch of data.

        Returns
        -------
        float
            The training loss.
        """
        tStart = time()
        
        batchData = self._getBatchData('train')
        sumOfLoss, nEffTerms = self.model.fit(batchData, self.step)
        # track the loss
        self._trainLoss.add(sumOfLoss, nEffTerms)

        tFinish = time()

        self._timePerStep.append(tFinish - tStart)

        if self.step and self.step % self.nStepsStatReport == 0:
            # if self.model.rank == 0:
            logger.info(("[STEP {0}] average number "
                         "of steps per second: {1:.1f}").format(
                self.step, 1. / np.average(self._timePerStep)))
            self._trainLogger.info(self._trainLoss.getAveLoss())
            logger.info("training loss: {0}".format(
                self._trainLoss.getAveLoss()))
            self._timePerStep = []
            self._trainLoss.reset()

    # @profile
    # @timeit
    def _validate(self, step):
        """
        Measures model validation performance.

        Returns
        -------
        dict
            A dictionary, where keys are the names of the loss metrics,
            and the values are the average value for that metric over
            the validation set.

        """
        if self._preloadValData == True:
            loss, predictions = self.model.validate(self._validationData)
            validScores = self._validationMetrics.update(
                predictions, self._allValidationTargets, step)
        else:
            losses = LossTracker()
            predictions = np.array([])
            allTargets = np.array([])

            if self._nValidationSamples is None:
                batchData = self._getBatchData(mode='validate')
                while batchData != None:
                    batchPreds, batchLoss, batchnEffTerms = self.model.batchValidate(batchData)
                    predictions = np.append(predictions, batchPreds.astype(np.float16))
                    allTargets = np.append(allTargets, batchData['targets'].astype(np.int8))
                    losses.add(batchLoss.item(), batchnEffTerms.item())
                    batchData = self._getBatchData(mode='validate')
            else:
                count = self.batchSize
                while count < self._nValidationSamples:
                    batchData = self._getBatchData(mode='validate')
                    batchPreds, batchLoss, batchnEffTerms = self.model.batchValidate(batchData)
                    predictions = np.append(predictions, batchPreds.astype(np.float16))
                    allTargets = np.append(allTargets, batchData['targets'].astype(np.int8))
                    losses.add(batchLoss.item(), batchnEffTerms.item())
                    count += self.batchSize
                remainder = self.batchSize - (count - self._nValidationSamples)
                batchSize = self.batchSize # Saving original batchSize
                self.batchSize = remainder # Changing batchSize to remainder
                batchData = self._getBatchData(mode='validate')
                batchPreds, batchLoss, batchnEffTerms = self.model.batchValidate(batchData)
                predictions = np.append(predictions, batchPreds.astype(np.float16))
                allTargets = np.append(allTargets, batchData['targets'].astype(np.int8))
                losses.add(batchLoss.item(), batchnEffTerms.item())
                self.batchSize = batchSize # Resetting batchSize to original
                

            # predictions = np.vstack(predictions)
            # allTargets = np.vstack(allTargets)
            predictions = predictions.reshape(-1, len(self.sampler._features))
            allTargets = allTargets.reshape(-1, len(self.sampler._features))
            loss = losses.getAveLoss()
            validScores = self._validationMetrics.update(
                predictions, allTargets, step)
            del predictions
            del allTargets

        # # if self.model.rank == 0:
        for name, score in validScores.items():
            logger.info("validation {0}: {1}".format(name, score))

        validScores["loss"] = loss

        to_log = [str(loss)]

        for k in sorted(self._validationMetrics.metrics.keys()):
            # logger.info("k: {0}".format(k))
            # if k in validScores and validScores[k].any():
            if k in validScores and validScores[k] is not None:
                to_log.append(str(validScores[k]))
            else:
                to_log.append("NA")
        self._validationLogger.info("\t".join(to_log))

        # scheduler update
        self.prev_lr = self.scheduler._last_lr[0]
        if self._useScheduler:
            if self.schedulerName == 'ReduceLROnPlateau':
                self.scheduler.step(math.ceil(loss * 1000.0) / 1000.0)
            else:
                self.scheduler.step()
            if self.prev_lr != self.scheduler._last_lr[0]:
                logger.info("UPDATES lr from {0} to: {1}".format(self.prev_lr, self.scheduler._last_lr[0]))
            else:
                logger.info("No change in learning rate {0}".format(self.scheduler._last_lr[0]))
            self.prev_lr = self.scheduler._last_lr[0]
        # save best_model
        if loss < self._minLoss:
            self._minLoss = loss
            self._saveCheckpoint({
                "step": self.step,
                "arch": self.model.__class__.__name__,
                "state_dict": self.model.getStateDict(),
                "min_loss": self._minLoss,
                "optimizer": self.model.getOptimizer().state_dict()}, True)
            logger.debug("Updating `best_model.pth.tar`")
        logger.info("validation loss: {0}".format(loss))




