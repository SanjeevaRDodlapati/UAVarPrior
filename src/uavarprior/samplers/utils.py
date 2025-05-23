'''
Created on May 6, 2021

@author: jsun
'''

import h5py as h5
import numpy as np
from collections import OrderedDict

EPS = 10e-8

def calcCWeight(targets, features, binarize_labels = None, valueOfMissing = None):
    '''
    Calculate class weight based on the class label distribution in the input
    parameter targets
    '''
    
    weights = {}
    for feat in features:
        target = targets[feat]
        if valueOfMissing is not None:
            target = target[target != valueOfMissing]
        if binarize_labels:
            target = (target >= 0.5).astype(np.uint8)
        if len(target) == 0:
            # no labeled examples, no weights to compute
            weights[feat] = {}
            continue
        
        uniLbls = np.unique(target)
        freq = np.zeros(len(uniLbls), dtype = np.int32)
        for iLbl in range(len(uniLbls)):
            freq[iLbl] = np.sum(target == uniLbls[iLbl])
        freq = freq / len(target) 

        w = 1 / freq
        w /= w.sum()
        w = OrderedDict(zip(uniLbls, w))
        weights[feat] = w
        
    return weights

def calcCWeightByH5File(h5Files, features = None,
                binarize_labels = None, valueOfMissing = None):
    '''
    Calculate class weight with data stored in H5 files
    '''
    with h5.File(h5Files[0], 'r') as fh:
        allFeats = [feat for feat in fh['outputs'].keys()]
    
    iFeatUse = None    
    if features is not None:
        if not all(feat in allFeats for feat in features):
            raise ValueError('Not all features can be found in h5 file')
        if len(features) < len(allFeats):
            # not every feature is needed
            iFeatUse = [i for i, feat in enumerate(allFeats) if feat in features]

    targets = {}
    for h5File in h5Files:
        with h5.File(h5File, 'r') as fh:
            ### collects outputs of each profile
            for feat in allFeats:
                if feat not in targets.keys():
                    targets[feat] = fh['outputs'][feat][:]
                else:
                    targets[feat] = np.concatenate((targets[feat], fh['outputs'][feat][:]))

    
    featsUse = allFeats
    if iFeatUse is not None:
        featsUse = [allFeats[i] for i in iFeatUse]    

    return calcCWeight(targets, featsUse, binarize_labels = binarize_labels,
                       valueOfMissing = valueOfMissing)

def getSWeight(targets, features = None, cWeights = None, 
               valueOfMissing = None):
    """Compute sample weights for model training.

    Computes sample weights given  a vector of output labels `y`. Sets weights
    of samples without label (`valueOfMissing`) to zero.

    Parameters
    ----------
    targets: :class:`numpy.ndarray`
        1d numpy array of output labels.
    cWeights: dict of dict
        Weight of target classes, e.g. peak presence, methylation states.

    Returns
    -------
    :class:`numpy.ndarray`
        Sample weights of size `targets`.
    """
    sWeights = np.ones(targets.shape, dtype = np.float32)
    if valueOfMissing is not None: 
        sWeights[targets == valueOfMissing] = 0
    if cWeights is not None:
        if features is None:
            raise ValueError('Parameter features cannot be None when '
                 'sample weights are created from class weights')
        for iFeat in range(len(features)):
            for lbl, weight in cWeights[features[iFeat]].items():
                sWeights[targets[:, iFeat] == lbl, iFeat] = weight
    return sWeights

