
from abc import ABCMeta
import os

from ....data import Genome
from ..seq_analyzer import SeqAnalyzer
from .utils import read_vcf_file

class GVarEvaluator(SeqAnalyzer, metaclass = ABCMeta):
    '''
    Base class of genetic variant impact evaluator

    variants are read in from vcf file
    
    Parameters
    -----------
    vcfFile : str
        Path to vcf File providing genetic variants to be evaluated.
        Must contain the columns: [#CHROM, POS, ID, REF, ALT], in order. 
        Column header does not need to be present.
    strandIdx: int or None, optional.
        Default is None. If applicable, specify the column index (0-based)
        in the VCF file that contains strand information for each variant.
    requireStrand : bool, optional.
        Default is False. Whether strand can be specified as '.'. If False,
        FuGEP accepts strand value to be '+', '-', or '.' and automatically
        treats '.' as '+'. If True, FuGEP skips any variant with strand '.'.
        This parameter assumes that `strandIdx` has been set.    
    '''
    
    # VARIANTEFFECT_COLS = ["chrom", "pos", "name", "ref", "alt", "strand", "ref_match", "contains_unk"]
    VARIANTEFFECT_COLS = ["chrom", "pos", "name"]

    def __init__(self, analysis, model, trainedModelPath, features,
                 vcfFile, model_built= 'pytorch', cpgFile = None, strandIdx = None, requireStrand = False,
                 outputDir = None, save_mult_pred = False,outputFormat = 'tsv',
                 seqLen = None, batchSize = 64, useCuda = False,
                 dataParallel = False, refSeq = Genome, genAssembly = None,
                 writeMemLimit = 5000, loggingVerbosity = 2):
        '''
        Construct a new object of 'GVarEvaluator'
        '''
        super(GVarEvaluator, self).__init__(model = model, 
                 trainedModelPath = trainedModelPath,
                model_built = model_built,
                 features = features, 
                 analysis = analysis, 
                 outputDir = outputDir,
                 save_mult_pred=save_mult_pred,
                 outputFormat = outputFormat,
                 seqLen = seqLen,
                 mode = 'varianteffect',
                 batchSize = batchSize, 
                 useCuda = useCuda,
                 dataParallel = dataParallel, 
                 refSeq = refSeq,
                 writeMemLimit = writeMemLimit,
                 loggingVerbosity = loggingVerbosity)
        
        self._vcfFile = vcfFile
        self._cpgFile = cpgFile
        self._strandIdx = strandIdx
        self._requireStrand = requireStrand
        
        # set up the output prefix
        path, filename = os.path.split(self._vcfFile)
        self._outputPathPrefix = '.'.join(filename.split('.')[:-1])
        if self._outputDir:
            os.makedirs(self._outputDir, exist_ok = True)
        else:
            self._outputDir = path
        self._outputPathPrefix = os.path.join(self._outputDir, self._outputPathPrefix)
        
        
        
