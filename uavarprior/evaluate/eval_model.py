"""
This module implements the ModelEvaluator class.
"""
import logging
import os
import warnings

import numpy as np
import torch

from ..utils import initialize_logger
from .metrics import PerformanceMetrics
from ..train import LossTracker



logger = logging.getLogger("fugep")


class ModelEvaluator(object):
    """
    Evaluate model on a test set of sequences with known targets.

    Parameters
    ----------
    model : torch.nn.Module
        The model architecture.
    lossCalculator : torch.nn._Loss
        The loss function that was optimized during training.
    dataSampler : fugep.samplers.Sampler
        Used to retrieve samples from the test set for evaluation.
    features : list(str)
        List of distinct features the model predicts.
    trainedModelPath : str
        Path to the trained model file, saved using `torch.save`.
    outputDir : str
        The output directory in which to save model evaluation and logs.
    batchSize : int, optional
        Default is 64. Specify the batch size to process examples.
        Should be a power of 2.
    nTestSamples : int or None, optional
        Default is `None`. Use `nTestSamples` if you want to limit the
        number of samples on which you evaluate your model. If you are
        using a sampler of type `fugep.samplers.OnlineSampler`,
        by default it will draw 640000 samples if `nTestSamples` is `None`.
    nMinMinorsReport : int, optional
        Default is 10. In the final test set, each class/feature must have
        more than `nMinMinorsReport` positive samples in order to
        be considered in the test performance computation. The output file that
        states each class' performance will report 'NA' for classes that do
        not have enough positive samples.
    useCuda : bool, optional
        Default is `False`. Specify whether a CUDA-enabled GPU is available
        for torch to use during training.
    dataParallel : bool, optional
        Default is `False`. Specify whether multiple GPUs are available
        for torch to use during training.
    useFeaturesOrd : list(str) or None, optional
        Default is None. Specify an ordered list of features for which to
        run the evaluation. The features in this list must be identical to or
        a subset of `features`, and in the order you want the resulting
        `test-targets.npz` and `testpredictions.npz` to be saved.
        TODO: Need a fix. This function seems not being currently implemented correctly, Javon, 05/26/2021

    Attributes
    ----------
    model : PredMWrapper
        The trained model.
    lossCalculator : torch.nn._Loss
        The model was trained using this loss function.
    sampler : fugep.samplers.Sampler
        The example generator.
    batchSize : int
        The batch size to process examples. Should be a power of 2.
    useCuda : bool
        If `True`, use a CUDA-enabled GPU. If `False`, use the CPU.
    dataParallel : bool
        Whether to use multiple GPUs or not.

    """

    def __init__(self,
                 model,
                 dataSampler,
                 trainedModelPath,
                 outputDir,
                 lossCalculator = None, # if None, lossCalculator is set to model directly
                 batchSize = 64,
                 nTestSamples = None,
                 nMinMinorsReport = 10,
                 useCuda=False,
                 dataParallel=False,
                 useFeaturesOrd=None,
                 valOfMisInTarget = None,
                 preloadTestData = False,
                 loggingVerbosity = 2):
        self.lossCalculator = lossCalculator
        self._preloadTestData = preloadTestData
        self.nTestSamples = nTestSamples

        self.model = model
        trainedModel = torch.load(
            trainedModelPath, map_location = lambda storage, location: storage)
        if "state_dict" in trainedModel:
            self.model.init(trainedModel["state_dict"])
        else:
            self.model.init(trainedModel)

        self.sampler = dataSampler
        if 'cWeights' in trainedModel:
            self.sampler.setClassWeights(trainedModel['cWeights'])

        self.outputDir = outputDir
        os.makedirs(outputDir, exist_ok = True)

        self.features = dataSampler.getFeatures()
        self._useIxs = list(range(len(self.features)))
        if useFeaturesOrd is not None:
            featureIxs = {f: ix for (ix, f) in enumerate(self.features)}
            self._useIxs = []
            self.features = []

            for f in useFeaturesOrd:
                if f in featureIxs:
                    self._useIxs.append(featureIxs[f])
                    self.features.append(f)
                else:
                    warnings.warn(("Feature {0} in `useFeaturesOrd` "
                                   "does not match any features in the list "
                                   "`features` and will be skipped.").format(f))
            self._saveFeaturesOrdered()

        initialize_logger(
            os.path.join(self.outputDir, "fugep.log"),
            verbosity = loggingVerbosity)

        self.dataParallel = dataParallel
        if self.dataParallel:
            self.model.toDataParallel()
            logger.debug("Wrapped model in DataParallel")

        self.useCuda = useCuda
        if self.useCuda:
            self.model.toUseCuda()

        self.batchSize = batchSize

        self._valOfMisInTarget = valOfMisInTarget
        self._metrics = PerformanceMetrics(
            self._getFeatureByIndex,
            nMinMinorsReport = nMinMinorsReport,
            valOfMisInTarget = self._valOfMisInTarget)

        if preloadTestData or nTestSamples:
            self._testData, self._allTestTargets = \
                self.sampler.getDataAndTargets(self.batchSize, nTestSamples)
            self._allTestTargets = self._allTestTargets[:, self._useIxs]
            # TODO: we should be able to do this on the sampler end instead of
            # here. the current workaround is problematic, since
            # self._testData still has the full featureset in it, and we
            # select the subset during `evaluate`
        

    def _saveFeaturesOrdered(self):
        """
        Write the feature ordering specified by `useFeaturesOrd`
        after matching it with the `features` list from the class
        initialization parameters.
        """
        fp = os.path.join(self.outputDir, 'use-features-ord.txt')
        with open(fp, 'w+') as fileHdl:
            for f in self.features:
                fileHdl.write('{0}\n'.format(f))

    def _getFeatureByIndex(self, index):
        """
        Gets the feature at an index in the features list.

        Parameters
        ----------
        index : int

        Returns
        -------
        str
            The name of the feature/target at the specified index.

        """
        return self.features[index]

    def _getBatchData(self, mode=None):
        """
        Fetches a mini-batch of examples

        Returns
        -------
        dict()
            A dictionary, the data contained are data type specific.
        """

        batchData = self.sampler.sample(
            batchSize = self.batchSize,
            mode = mode)

        return batchData

    def evaluate(self):
        """
        Passes all samples retrieved from the sampler to the model in
        batches and returns the predictions. Also reports the model's
        performance on these examples.

        Returns
        -------
        dict
            A dictionary, where keys are the features and the values are
            each a dict of the performance metrics (currently ROC AUC and
            AUPR) reported for each feature the model predicts.

        """
        if self._preloadTestData == True:
            loss, predictions = self.model.validate(self._testData)

        else:
            losses = LossTracker()
            predictions = np.array([])
            allTargets = np.array([])
            if self.nTestSamples is None:
                batchData = self._getBatchData(mode='test')
                count = self.batchSize
                while batchData != None:
                    batchPreds, batchLoss, batchnEffTerms = self.model.batchValidate(batchData)
                    predictions = np.append(predictions, batchPreds.astype(np.float16))
                    allTargets = np.append(allTargets, batchData['targets'].astype(np.int8))
                    losses.add(batchLoss.item(), batchnEffTerms.item())
                    batchData = self._getBatchData(mode='test')
                    count += self.batchSize
                    if (count//self.batchSize)%1000 == 0:
                        logger.info("Number of samples evaluated: {}".format(count))
                logger.info("Total number of samples evaluated: {}".format(count))
            else:
                batchData = self._getBatchData(mode='test')
                count = self.batchSize
                while (count < self.nTestSamples) & (batchData != None):
                    batchPreds, batchLoss, batchnEffTerms = self.model.batchValidate(batchData)
                    predictions = np.append(predictions, batchPreds.astype(np.float16))
                    allTargets = np.append(allTargets, batchData['targets'].astype(np.int8))
                    losses.add(batchLoss.item(), batchnEffTerms.item())
                    batchData = self._getBatchData(mode='test')
                    count += self.batchSize
                    if (count//self.batchSize)%1000 == 0:
                        logger.info("Number of samples evaluated: {}".format(count))
                logger.info("Total number of samples evaluated: {}".format(count))
                # remainder = self.batchSize - (count - self.nTestSamples)
                # batchSize = self.batchSize  # Saving original batchSize
                # self.batchSize = remainder  # Changing batchSize to remainder
                # batchData = self._getBatchData(mode='test')
                # batchPreds, batchLoss, batchnEffTerms = self.model.batchValidate(batchData)
                # predictions = np.append(predictions, batchPreds.astype(np.float16))
                # allTargets = np.append(allTargets, batchData['targets'].astype(np.int8))
                # losses.add(batchLoss.item(), batchnEffTerms.item())
                # self.batchSize = batchSize  # Resetting batchSize to original
                

            predictions = np.vstack(predictions)
            allTargets = np.vstack(allTargets)
            self._allTestTargets = allTargets
            loss = losses.getAveLoss()

        average_scores = self._metrics.update(
                predictions, self._allTestTargets)

        #self._metrics.visualize(predictions, self._allTestTargets, self.outputDir)

        np.savez_compressed(
            os.path.join(self.outputDir, "test-predictions.npz"),
            data = predictions)

        np.savez_compressed(
            os.path.join(self.outputDir, "test-targets.npz"),
            data=self._allTestTargets)

        logger.info("test loss: {0}".format(loss))
        for name, score in average_scores.items():
            logger.info("test {0}: {1}".format(name, score))

        test_performance = os.path.join(
            self.outputDir, "test-performance.txt")
        feature_scores_dict = self._metrics.write_feature_scores_to_file(
            test_performance)

        return feature_scores_dict
