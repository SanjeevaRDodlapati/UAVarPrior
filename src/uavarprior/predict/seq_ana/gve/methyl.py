'''
@author: Sanjeeva Reddy Dodlapati

This code is to evaluate effect of genetic variant on CpG methylation prediction
'''

from time import time
import logging
import math
import numpy as np
import tensorflow as tf

from ....data import Genome
from ....data.h5.methylH5Data import readCpgProf
from ...utils import get_reverse_complement_encoding
from ...utils import _truncate_sequence

from .utils import read_vcf_file
from .gve_evaluator import GVarEvaluator

logger = logging.getLogger("uavarprior")

class MethylVarEvaluator(GVarEvaluator):
    '''
    Implementation of variant effect evaluator by 
    applying a model trained for predicting peak type events
    
    Parameters
    -----------
    vcfFile : str
        Path to vcf File providing genetic variants to be evaluated.
        Must contain the columns: [#CHROM, POS, ID, REF, ALT], in order. 
        Column header does not need to be present.
    strandIdx: int or None, optional.
        Default is None. If applicable, specify the column index (0-based)
        in the VCF file that contains strand information for each variant.
    '''


    def __init__(self, analysis, model, trainedModelPath, features,
                 vcfFile, cpgFile, model_built= 'pytorch', strandIdx = None, requireStrand = False,
                 outputDir = None, save_mult_pred = False, outputFormat = 'tsv',
                 seqLen = None, batchSize = 64, useCuda = False,
                 dataParallel = False, refSeq = Genome, genAssembly = None,
                 writeMemLimit = 5000, loggingVerbosity = 2):
        '''
        Construct a new object of 'GVarEvaluator'
        '''
        super(MethylVarEvaluator, self).__init__(model = model,
                 trainedModelPath = trainedModelPath,
                model_built = model_built,
                 features = features, 
                 analysis = analysis, 
                 vcfFile = vcfFile,
                 cpgFile = cpgFile,
                 strandIdx = strandIdx,
                 requireStrand = requireStrand,
                 outputDir = outputDir,
                 save_mult_pred=save_mult_pred,
                 outputFormat = outputFormat,
                 seqLen = seqLen,
                 batchSize = batchSize, 
                 useCuda = useCuda,
                 dataParallel = dataParallel, 
                 refSeq = refSeq,
                 genAssembly = genAssembly,
                 writeMemLimit = writeMemLimit,
                 loggingVerbosity = loggingVerbosity)
        
        self._genAssembly = genAssembly
        if self._genAssembly == 'ncbi':
            self._change_chrom(genAssembly)

        # load variants
        self._variants = read_vcf_file(self._vcfFile, strand_index=self._strandIdx,
                                       require_strand=self._requireStrand,
                                       output_NAs_to_file="{0}-invalid.vcf".format(self._outputPathPrefix),
                                       seq_context=(self._startRadius, self._endRadius),
                                       reference_sequence=self._refSeq)
        # load cpg
        self._cpgProf = readCpgProf(self._cpgFile)
        self._reporters = self._initializeReporters(self._outputPathPrefix,
                                                    ["chrom", "cpg_pos", "var_pos", "var_name", "ref", "alt", "strand", "match", "unk"],
                                                    self._model._mult_predictions, save_mult_pred, outputSize=len(self._variants),
                                                    outputFormat=self._outputFormat)

    def _change_chrom(self, genAssembly):
        if genAssembly == 'ncbi':
            chrom_map = {'NC_000001.11': '1', 'NC_000002.12': '2', 'NC_000003.12': '3', 'NC_000004.12': '4',
                         'NC_000005.10': '5',
                         'NC_000006.12': '6', 'NC_000007.14': '7', 'NC_000008.11': '8', 'NC_000009.12': '9',
                         'NC_000010.11': '10',
                         'NC_000011.10': '11', 'NC_000012.12': '12', 'NC_000013.11': '13', 'NC_000014.9': '14',
                         'NC_000015.10': '15',
                         'NC_000016.10': '16', 'NC_000017.11': '17', 'NC_000018.10': '18', 'NC_000019.10': '19',
                         'NC_000020.11': '20',
                         'NC_000021.9': '21', 'NC_000022.11': '22', 'NC_000023.11': '23', 'NC_000024.10': '24',
                         'NC_012920.1': 'MT'}

            for k, v in chrom_map.items():
                self._refSeq.genome.records[v] = self._refSeq.genome.records.pop(k)
            self._refSeq.chrs = list(self._refSeq.genome.records.keys())
            self._refSeq.len_chrs = self._refSeq._get_len_chrs()


    def _getRefIdxs(self, refLen):
        '''
        Assume the reference is centered in the sequence
        '''
        mid = self._seqLen // 2
        if self._seqLen % 2 == 0:
            mid -= 1
        startPos = mid - refLen // 2
        endPos = startPos + refLen
        return (startPos, endPos)
    
    
    def _processAlt(self, chrom, pos, ref, alt, start, end, refSeqEnc, strand = '+'):
        """
        Return the encoded sequence centered at a given allele for input into
        the model.
    
        Parameters
        ----------
        chrom : str
            The chromosome the variant is in
        pos : int
            The position of the variant
        ref : str
            The reference allele of the variant
        alt : str
            The alternate allele
        start : int
            The start coordinate of reference squence (refSeqEnc) in genome 
        end : int
            The end coordinate of reference squence (refSeqEnc) in genome 
        refSeqEnc : numpy.ndarray
            The reference sequence encoding
            It is assumed the refSeq comes from positive strand
        strand : strand of the variant
        
        Returns
        -------
        list(numpy.ndarray)
            A list of the encoded sequences containing alternate alleles at
            the center
    
        """
        if alt == '*' or alt == '-':   # indicates a deletion
            alt = ''
        refLen = len(ref)
        altLen = len(alt)
        if altLen > len(refSeqEnc):
            sequence = _truncate_sequence(alt, len(refSeqEnc))
            return self._refSeq.sequence_to_encoding(sequence)
    
        altEnc = self._refSeq.sequence_to_encoding(alt)
        if strand == '-':
            altEnc = self._refSeq.getComplementEncoding(altEnc)
        
        if refLen == altLen:  # substitution
            startPos, endPos = self._getRefIdxs(refLen)
            sequence = np.vstack([refSeqEnc[:startPos, :], altEnc, refSeqEnc[endPos:, :]])
            return sequence
        elif altLen > refLen:  # insertion
            startPos, endPos = self._getRefIdxs(refLen)
            sequence = np.vstack([refSeqEnc[:startPos, :], altEnc, refSeqEnc[endPos:, :]])
            truncStart = (len(sequence) - refSeqEnc.shape[0]) // 2
            truncEnd = truncStart + refSeqEnc.shape[0]
            sequence = sequence[truncStart:truncEnd, :]
            return sequence
        else:  # deletion
            lhs = self._refSeq.get_sequence_from_coords(chrom,
                start - refLen // 2 + altLen // 2,
                pos + 1, pad = True)
            rhs = self._refSeq.get_sequence_from_coords(chrom, pos + 1 + refLen,
                end + math.ceil(refLen / 2.) - math.ceil(altLen / 2.),
                pad = True)
            sequence = lhs + alt + rhs
            if not len(sequence) == len(refSeqEnc):
                # print(f'Lenth of AltSequence mismatch with the reference sequence')
                if len(lhs) > len(rhs):
                    sequence = sequence[len(sequence)-len(refSeqEnc): len(sequence)]
                else:
                    sequence = sequence[:len(refSeqEnc)]
            return self._refSeq.sequence_to_encoding(sequence)
    
    
    def _handleStandardRef(self,refPos, refEnc, seqEnc):
        # if reference allele encoding in seqEnc (from genome) 
        # does not match refEnc (from vcf), refEnc is used
        refLen = refEnc.shape[0]
        # startPos, _ = self._getRefIdxs(refLen)
    
        seqEncAtRef = seqEnc[refPos:refPos + refLen, :]
        match = np.array_equal(seqEncAtRef, refEnc)
    
        seqAtRef = None
        if not match:
            print(f'refEnc, and seqAtRef are not matching')
            seqAtRef = self._refSeq.encoding_to_sequence(seqEncAtRef)
            seqEnc[refPos:refPos + refLen, :] = refEnc # replace
        return match, seqEnc, seqAtRef
    
    
    def _handleLongRef(self, refEnc, seqEnc):
        refLen = refEnc.shape[0]
        seqEncAtRef = seqEnc
        refStart = refLen // 2 - self._startRadius - 1
        refEnd = refLen // 2 + self._endRadius - 1;[]
        refEnc = refEnc[refStart:refEnd]
        match = np.array_equal(seqEncAtRef, refEnc)
    
        seqRef = None
        if not match:
            seqRef = self._refSeq.encoding_to_sequence(seqEncAtRef)
            seqEnc = refEnc # use ref in vcf
        return match, seqEnc, seqRef
    
    
    def _handleRefAltPredictions(self, batchRefSeqs, batchAltSeqs, batchIds, model_built=None):
        """
        Helper method for variant effect prediction. Gets the model
        predictions and updates the reporters.
    
        Parameters
        ----------
        batchRefSeqs : list(np.ndarray)
            One-hot encoded sequences with the ref base(s).
        batchAltSeqs : list(np.ndarray)
            One-hot encoded sequences with the alt base(s).
            
        Returns
        -------
        None
    
        """
        batchRefSeqs = np.array(batchRefSeqs)
        batchAltSeqs = np.array(batchAltSeqs)
        if self._model._model_built == 'tensorflow':
            batchRefSeqs = tf.convert_to_tensor(batchRefSeqs)
            batchAltSeqs = tf.convert_to_tensor(batchAltSeqs)
        n_pred = self._model._mult_predictions
        if n_pred > 1:
            refOutputs = self._model.predict_mult([{'sequence': batchRefSeqs}])
            altOutputs = self._model.predict_mult([{'sequence': batchAltSeqs}])
            for r in self._reporters:
                if r.needs_base_pred:
                    r.handle_batch_mult_predictions_temp(altOutputs, batchIds, refOutputs)
                else:
                    r.handle_batch_mult_predictions(altOutputs, batchIds)
        else:
            refOutputs = self._model.predict([{'sequence': batchRefSeqs}])
            altOutputs = self._model.predict([{'sequence': batchAltSeqs}])
            for r in self._reporters:
                if r.needs_base_pred:
                    r.handle_batch_predictions(altOutputs, batchIds, refOutputs)
                else:
                    r.handle_batch_predictions(altOutputs, batchIds)
            # except:
            #     print(f'batchRefSeqs: {batchRefSeqs.shape}')
            #     print(f'batchAltSeqs: {batchAltSeqs.shape}')

        # else:
        #     refOutputs = self._model.predict([{'sequence': batchRefSeqs}])
        #     altOutputs = self._model.predict([{'sequence': batchAltSeqs}])
        #     for r in self._reporters:
        #         if r.needs_base_pred:
        #             r.handle_batch_predictions(altOutputs, batchIds, refOutputs)
        #         else:
        #             r.handle_batch_predictions(altOutputs, batchIds)



        
    def evaluate(self, inputData = None):
        """
        Get model predictions and scores for a list of variants.

        Parameters
        ----------
        inputData : dummy input for compatibility of the base class
            Genetic variants to evaluate come from self._vcfFile

        Returns
        -------
        None
            Saves all files to `self._outputDir`. If any bases in the 'ref' column
            of the VCF do not match those at the specified position in the
            reference genome, the row labels .txt file will mark this variant
            as `ref_match = False`. If most of your variants do not match
            the reference genome, please check that the reference genome
            you specified matches the version with which the variants were
            called. The predictions can used directly if you have verified that
            the 'ref' bases specified for these variants are correct (Selene
            will have substituted these bases for those in the reference
            genome). In addition, if any base in the retrieved reference
            sequence is unknown, the row labels .txt file will mark this variant
            as `contains_unk = True`. Finally, some variants may show up in an
            'NA' file. This is because the surrounding sequence context ended up
            being out of bounds or overlapping with blacklist regions  or the
            chromosome containing the variant did not show up in the reference
            genome FASTA file.

        """
        startTime = time()
        num_variants = len(self._variants)
        cpgProf = self._cpgProf
        batchRefSeqs, batchAltSeqs, batchIds = [], [], []
        stepTime = time()
        for ix, (chrom, pos, name, ref, alt, strand) in enumerate(self._variants):
            # centers the sequence containing the ref allele based on the size
            # of ref
            center = pos + len(ref) // 2
            start = center - self._startRadius #- len(ref) // 2
            end = center + self._endRadius # + math.ceil(len(ref) / 2.)
            chr_cpg = cpgProf[cpgProf.chrom==chrom]
            if self._genAssembly == 'ncbi':
                chr_cpg_pos = chr_cpg.pos+1
            else:
                chr_cpg_pos = chr_cpg.pos

            ind = (chr_cpg_pos < end- len(ref) // 2) & (chr_cpg_pos > start+ len(ref) // 2)
            win_cpg = chr_cpg_pos[ind]
            if win_cpg.shape[0] > 0:
                for cpg in win_cpg:
                    cpg_win_start = cpg - self._startRadius
                    cpg_win_end = cpg + self._endRadius
                    # if pos < cpg_win_start-1 or pos > cpg_win_end-1:
                    #     print(f'Pos {pos} out of cpg window: {cpg_win_start-1}-{cpg_win_end-1}  refLen: {len(ref)}, centere: {center}')
                    #     pass
                    # else:
                    refSeqEnc, containsUnk = self._refSeq.get_encoding_from_coords_check_unk(chrom, cpg_win_start-1, cpg_win_end-1)

                    refEnc = self._refSeq.sequence_to_encoding(ref)
                    if strand == '-':
                        refEnc = self._refSeq.getComplementEncoding(refEnc)
                    altSeqEnc = self._processAlt(chrom, pos, ref, alt, cpg_win_start-1, cpg_win_end-1, refSeqEnc)

                    # check if the reference sequence of the variant matches with reference genome
                    match = True
                    seqAtRef = None
                    if len(ref) and len(ref) < self._seqLen and (pos-cpg_win_start) > 0:
                        match, refSeqEnc, seqAtRef = self._handleStandardRef(pos-cpg_win_start, refEnc, refSeqEnc)
                    elif len(ref) >= self._seqLen:
                        match, refSeqEnc, seqAtRef = self._handleLongRef(refEnc, refSeqEnc)

                    if containsUnk:
                        logger.warn("For variant ({0}, {1}, {2}, {3}, {4}, {5}), "
                                   "reference sequence contains unknown base(s)"
                                   "--will be marked `True` in the `contains_unk` column "
                                   "of the .tsv or the row_labels .txt file.".format(
                                     chrom, pos, name, ref, alt, strand))
                    if not match:
                        logger.warn("For variant ({0}, {1}, {2}, {3}, {4}, {5}), "
                                      "reference does not match the reference genome. "
                                      "Reference genome contains {6} instead. "
                                      "Predictions/scores associated with this "
                                      "variant--where we use '{3}' in the input "
                                      "sequence--will be marked `False` in the `ref_match` "
                                      "column of the .tsv or the row_labels .txt file".format(
                                          chrom, pos, name, ref, alt, strand, seqAtRef))

                    batchIds.append((chrom, cpg, pos, name, ref, alt, strand, match, containsUnk))
                    if strand == '-':
                        refSeqEnc = get_reverse_complement_encoding(refSeqEnc,
                            self._refSeq.BASES_ARR, self._refSeq.COMPLEMENTARY_BASE_DICT)
                        altSeqEnc = get_reverse_complement_encoding(altSeqEnc,
                            self._refSeq.BASES_ARR, self._refSeq.COMPLEMENTARY_BASE_DICT)
                    # if not refSeqEnc.shape == altSeqEnc.shape:
                    #     print(f'refSeqEnc and altSeqEnc do not have the same shape')
                    batchRefSeqs.append(refSeqEnc)
                    batchAltSeqs.append(altSeqEnc)
                    # else:
                    #     print(f'refSeqEnc and altSeqEnc do not have the same shape')

                if len(batchRefSeqs) >= self._batchSize:
                    self._handleRefAltPredictions(batchRefSeqs, batchAltSeqs, batchIds)
                    batchRefSeqs, batchAltSeqs, batchIds = [], [], []

                if ix and ix % 1000 == 0:
                    print("[STEP {0}/{1}]: {2} s to process 1000 variants. ".format(
                        ix, num_variants, time() - stepTime))
                    stepTime = time()

            else:
                print(f'No cpg for index: {ix}, vcf pos: {pos}')

        if batchRefSeqs:
            self._handleRefAltPredictions(batchRefSeqs, batchAltSeqs, batchIds)
            batchRefSeqs, batchAltSeqs, batchIds = [], [], []


        for r in self._reporters:
            r.write_to_file()
