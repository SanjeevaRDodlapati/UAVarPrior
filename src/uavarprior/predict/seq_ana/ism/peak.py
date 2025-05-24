'''
Created on May 28, 2021

@author: Sanjeeva Reddy Dodlapati
'''

import numpy as np
import os
import pyfaidx
import logging

from .ism_analyzer import ISMSeqAnalyzer
from .utils import generateMutation
from .utils import mutateSeqEnc
from .utils import _mutationId
from ....data import Genome
from curses.ascii import isdigit

logger = logging.getLogger("uavarprior")

class PeakISMSeqAnalyzer(ISMSeqAnalyzer):
    '''
    In-silico mutagenesis sequence analyzer using 
    models for predicting peak type events, e.g., chromatin
    accessibility
    
    Parameters
    -----------
    outputFormat : {'tsv', 'hdf5'}
            The desired output format.
    nMutBase : int, optional
        Default is 1. The number of bases to mutate at one time in
        *in silico* mutagenesis.
    startPosition : int, optional
        Default is 0. The starting position of the subsequence to be
        mutated.
    endPosition : int or None, optional
        Default is None. The ending position of the subsequence to be
        mutated. If left as `None`, then `self._seqLen` will be
        used.        
    '''

    def __init__(self, analysis, model, trainedModelPath, 
                 features, outputDir = None, outputFormat = 'tsv', 
                 seqLen = None, batchSize = 64, useCuda = False,
                 dataParallel = False, refSeq = Genome,
                 writeMemLimit = 5000, loggingVerbosity = 2, 
                 nMutBase = 1, startPosition = 0, endPosition = None):
        '''
        Construct a new object of PeakISMSeqAnalyzer
        
        Raises
        ------
        ValueError
            If the value of `startPosition` or `endPosition` is negative.
        ValueError
            If there are fewer than `nMutBase` between `startPosition`
            and `endPosition`.
        ValueError
            If `startPosition` is greater or equal to `endPosition`.
        ValueError
            If `startPosition` is not less than `self._seqLen`.
        ValueError
            If `endPosition` is greater than `self._seqLen`.
        '''
        
        super(PeakISMSeqAnalyzer, self).__init__(
            model = model, 
            trainedModelPath = trainedModelPath,
            features = features, 
            analysis = analysis, 
            nMutBase = nMutBase,
            outputDir = outputDir,
            outputFormat = outputFormat,
            seqLen = seqLen,
            batchSize = batchSize, 
            useCuda = useCuda,
            dataParallel = dataParallel, 
            refSeq = refSeq,
            writeMemLimit = writeMemLimit,
            loggingVerbosity = loggingVerbosity)
        
        self._startPos = startPosition
        self._endPos = endPosition
        if self._endPos is None:
            self._endPos = self._seqLen
        self._validateSeqPosition()
        
    def _predictForMutation(self, sequence, refPreds, mutations, reporters = []):
        
        """
        Get the predictions for all specified mutations applied
        to a given sequence and, if applicable, compute the scores
        ("abs_diffs", "diffs", "logits") for these mutations.

        Parameters
        ----------
        sequence : str
            The sequence to mutate.
        refPreds : numpy.ndarray
            The model's prediction for `sequence`.
        mutations : list(list(tuple))
            The mutations to apply to the sequence. Each element in
            `mutations` is a list of tuples, where each tuple
            specifies the `int` position in the sequence to mutate and what
            `str` base to which the position is mutated (e.g. (1, 'A')).
        reporters : list(PredictionsHandler)
            The list of reporters, where each reporter handles the predictions
            made for each mutated sequence. Will collect, compute scores
            (e.g. `AbsDiffScoreHandler` computes the absolute difference
            between `refPreds` and the predictions for the mutated
            sequence), and output these as a file at the end.

        Returns
        -------
        None
            Writes results to files corresponding to each reporter in
            `reporters`.
        
        Note
        -------
        Modified from Selene's in_silico_mutagenesis_predict
        """
        
        refSeqEnc = self._refSeq.sequence_to_encoding(sequence)
        for i in range(0, len(mutations), self._batchSize):
            start = i
            end = min(i + self._batchSize, len(mutations))
            mutSeqEnc = np.zeros((end - start, *refSeqEnc.shape))

            batchIds = []
            for ix, mutation in enumerate(mutations[start:end]):
                mutSeqEnc[ix, :, :] = mutateSeqEnc(refSeqEnc, mutation,
                    refSeq=self._refSeq)
                batchIds.append(_mutationId(sequence, mutation))
            outputs = self._model.predict([{'sequence': mutSeqEnc}])

            for r in reporters:
                if r.needs_base_pred:
                    r.handle_batch_predictions(outputs, batchIds, refPreds)
                else:
                    r.handle_batch_predictions(outputs, batchIds)

        for r in reporters:
            r.write_to_file()
    
    def _validateSeqPosition(self):
        if self._startPos >= self._endPos:
                raise ValueError(("Starting positions must be less than the ending "
                                  "positions. Found a starting position of {0} with "
                                  "an ending position of {1}.").format(self._startPos,
                                                                       self._endPos))
        if self._startPos < 0:
            raise ValueError("Negative starting positions are not supported.")
        if self._endPos < 0:
            raise ValueError("Negative ending positions are not supported.")
        if self._startPos >= self._seqLen:
            raise ValueError(("Starting positions must be less than the sequence length."
                              " Found a starting position of {0} with a sequence length "
                              "of {1}.").format(self._startPos, self._seqLen))
        if self._endPos > self._seqLen:
            raise ValueError(("Ending positions must be less than or equal to the sequence "
                              "length. Found an ending position of {0} with a sequence "
                              "length of {1}.").format(self._endPos, self._seqLen))
        if (self._endPos - self._startPos) < self._nMutBase:
            raise ValueError(("Fewer bases exist in the substring specified by the starting "
                              "and ending positions than need to be mutated. There are only "
                              "{0} currently, but {1} bases must be mutated at a "
                              "time").format(self._endPos - self._startPos, self._nMutBase))
    
    def _evaluate(self, sequence, outputPathPrefix, reportRefPred = True):
        """
        Applies *in silico* mutagenesis to a sequence.

        Parameters
        ----------
        sequence : str
            The sequence to mutate.
        outputPathPrefix : str
            The path to which the data files are written. If directories in
            the path do not yet exist they will be automatically created.
        reportRefPred: bool, optional.
            Default is True
            If True, no report of the prediction for the reference 
        
        Returns
        -------
        None
            Outputs data files from *in silico* mutagenesis to `self._outputDir`.
            For HDF5 output and 'predictions' in `self._analysis`, an additional
            file named `*_ref_predictions.h5` will be outputted with the
            model prediction for the original input sequence.

        Note
        ------
        Modified from Selene's in_silico_mutagenesis function
    
        """
        
        sequence = self._padOrTruncateSeq(sequence)
        mutations = generateMutation(sequence, nMutBase = self._nMutBase,
            refSeq = self._refSeq, startPosition = self._startPos,
            endPosition = self._endPos)
        reporters = self._initializeReporters(outputPathPrefix, self.ISM_COLS, 
            outputSize = len(mutations), outputFormat = self._outputFormat)
        
        refSeqEnc = self._refSeq.sequence_to_encoding(sequence)
        refSeqEnc = refSeqEnc.reshape((1, *refSeqEnc.shape))
        refPreds = self._model.predict([{'sequence': refSeqEnc}])
               
        if reportRefPred:
            if "predictions" in self._analysis and self._outputFormat == 'hdf5':
                if os.path.isdir(outputPathPrefix):
                    refRepPath = os.path.join(outputPathPrefix, 'ref')
                else:
                    refRepPath = "{0}-ref".format(outputPathPrefix)
                refReporter = self._initializeReporters(refRepPath,
                    ["name"], outputSize = 1, outputFormat = self._outputFormat, 
                    analysis = ["predictions"])[0]
                refReporter.handle_batch_predictions(refPreds, [["input_sequence"]])
                refReporter.write_to_file()
            elif "predictions" in self._analysis and self._outputFormat == 'tsv':
                reporters[-1].handle_batch_predictions(
                    refPreds, [["input_sequence", "NA", "NA"]])

        self._predictForMutation(sequence, refPreds, mutations, reporters = reporters)
        
        return refPreds
    
    def _evaluateForFastaFile(self, filePath, useSeqName):
        '''
        Evaluate sequences in fasta file
        '''
        
        fastaFile = pyfaidx.Fasta(filePath)
        for i, fasta_record in enumerate(fastaFile):
            self._evaluate(str.upper(str(fasta_record)), )
            seq = self._padOrTruncateSeq(str.upper(str(fasta_record)))
            filePrefix = None
            if useSeqName:
                filePrefix = os.path.join(
                    self._outputDir, fasta_record.name.replace(' ', '_'))
            else:
                filePrefix = os.path.join(self._outputDir, str(i))
            self._evaluate(seq, filePrefix)
        
        fastaFile.close()
        
    def _evaluateForBedFile(self, filePath, strandIdx, useSeqPos):
        '''
        Evaluate sequences in the bed file
        
        Parameter
        ----------
        useSeqPos: bool
            If True, the position(chr_start_end) will appear in the output file name,
            otherwise just the index 
        '''
        # retrieve sequence coordinates
        seqCoords, labels = self._getSeqFromBedFile(filePath, 
                self._outputDir, strandIdx = strandIdx)
        refReporter = self._initializeReporters(
            os.path.join(self._outputDir, 'ref'),
            ["index", "chrom", "start", "end", "strand", "contains_unk"],
            outputSize = len(labels), outputFormat = self._outputFormat,
            analysis = ['predictions'])[0]
            
        for _, (label, coords) in enumerate(zip(labels, seqCoords)):
            seq = self._refSeq.get_sequence_from_coords(*coords, pad = True)
            containsUnk = self._refSeq.UNK_BASE in seq
            if useSeqPos:
                filePrefix = os.path.join(self._outputDir, '_'.join(
                    ['chr' + coords[0] if isdigit(coords[0]) else coords[0], 
                     str(coords[1]), str(coords[2])]))
            else:
                filePrefix = os.path.join(self._outputDir, 'seq' + str(label[0]))
            
            refPred = self._evaluate(seq, filePrefix, reportRefPred = False)
            refReporter.handle_batch_predictions(refPred, [label + (containsUnk,)])
            
            if containsUnk:
                logger.warn(("For region {0}, "
                               "reference sequence contains unknown "
                               "base(s). --will be marked `True` in the "
                               "`contains_unk` column of the .tsv or "
                               "row_labels .txt file.").format(label))
                
        refReporter.write_to_file()
        
    
    def evaluate(self, inputData, strandIdx = None, useSeqName = True, useSeqPos = True):
        """
        Apply *in silico* mutagenesis to all sequences in a FASTA file.

        Please note that we have not parallelized this function yet, so runtime
        increases exponentially when you increase `nMutBase`.

        Parameters
        ----------
        inputData: str or dict()
            A single sequence, or a path to the FASTA or BED file input.
        save_data : list(str)
            A list of the data files to output. Must input 1 or more of the
            following options: ["abs_diffs", "diffs", "logits", "predictions"].
        useSeqName : bool, optional.
            Default is True. Only effective when evaluating sequences in fasta file
            If `useSeqName`, output files are prefixed
            by the sequence name/description corresponding to each sequence
            in the FASTA file. Spaces in the sequence name are replaced with
            underscores '_'. If not `useSeqName`, output files are
            prefixed with an index :math:`i` (starting with 0) corresponding
            to the :math:`i`th sequence in the FASTA file.
        strandIdx : int or None, optional
            Default is None. If the trained model makes strand-specific
            predictions, your input BED file may include a column with strand
            information (strand must be one of {'+', '-', '.'}). Specify
            the index (0-based) to use it. Otherwise, by default '+' is used.
            (This parameter is ignored if FASTA file is used as input.)
        useSeqPos : bool, optional.
            Default is True. Only effective when evaluating sequences in bed file
            If `useSeqPos`, output files are prefixed
            by the sequence position(chr_start_end) corresponding to each sequence
            in the bed file. If not `useSeqPos`, output files are
            prefixed with an index :math:`i` (starting with 0) corresponding
            to the :math:`i`th sequence in the bed file.
        

        Returns
        -------
        None
            Outputs data files from *in silico* mutagenesis to `self._outputDir`.
            For HDF5 output and 'predictions' in `self._analysis`, an additional
            file named `*-ref-predictions.h5` [predicting for fasta file] 
            (or ref-predicitons.h5 [predicting for bed file]) will be outputted with the
            model prediction for the original input sequence.

        Notes:
        If evaluate for a file, each sequence in the file will have its own 
        set of output files, where the number of output files depends on the 
        number of `self._analysis` predictions/scores specified.
        """
        
        if isinstance(inputData, str) and \
            not(inputData.endswith('.fa') or inputData.endswith('.fasta') or \
                inputData.endswith('.bed')):
            raise ValueError('Unrecognized input file: {0}'.format(inputData))
        
        os.makedirs(self._outputDir, exist_ok = True)
        if inputData.endswith('fasta') or inputData.endswith('fa'):
            self._evaluateForFastaFile(inputData, useSeqName)
        elif inputData.endswith('bed'):
            self._evaluateForBedFile(inputData, strandIdx, useSeqPos)
        else:
            # a dictionary contains key 'sequence' providing sequence is assumed
            self._evaluate(inputData['sequence'], self._outputDir)

        