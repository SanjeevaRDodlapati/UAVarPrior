import itertools

import numpy as np

from ....data import Genome


def generateMutation(sequence, nMutBase = 1, refSeq = Genome,
            startPosition = 0, endPosition = None):
    """
    Creates a list containing each mutation that occurs from an
    *in silico* mutagenesis across the whole sequence.

    Please note that we have not parallelized this function yet, so
    runtime increases exponentially when you increase `nMutBase`.

    Parameters
    ----------
    sequence : str
        A string containing the sequence we would like to mutate.
    nMutBase : int, optional
        Default is 1. The number of base changes to make with each set of
        mutations evaluated, e.g. `nMutBase = 2` considers all
        pairs of SNPs.
    refSeq : class, optional
        Default is `uavarprior.data.sequences.Genome`. The type of sequence
        that has been passed in.
    startPosition : int, optional
        Default is 0. The starting position of the subsequence to be
        mutated.
    endPosition : int or None, optional
        Default is None. The ending position of the subsequence to be
        mutated. If left as `None`, then `len(sequence)` will be
        used.

    Returns
    -------
    list(list(tuple))
        A list of all possible mutations. Each element in the list is
        itself a list of tuples, e.g. element = [(0, 'T')] when only mutating
        1 base at a time. Each tuple is the position to mutate and the base
        with which we are replacing the reference base.

        For a sequence of length 1000, mutating 1 base at a time means that
        we return a list with length of 3000-4000, depending on the number of
        unknown bases in the input sequences.

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
        If `startPosition` is not less than `len(sequence)`.
    ValueError
        If `endPosition` is greater than `len(sequence)`.

    Note
    -------
    Modified from Selene' in_silico_mutagenesis_sequences function
    """
    if endPosition is None:
        endPosition = len(sequence)
    if startPosition >= endPosition:
        raise ValueError(("Starting positions must be less than the ending "
                          "positions. Found a starting position of {0} with "
                          "an ending position of {1}.").format(startPosition,
                                                               endPosition))
    if startPosition < 0:
        raise ValueError("Negative starting positions are not supported.")
    if endPosition < 0:
        raise ValueError("Negative ending positions are not supported.")
    if startPosition >= len(sequence):
        raise ValueError(("Starting positions must be less than the sequence length."
                          " Found a starting position of {0} with a sequence length "
                          "of {1}.").format(startPosition, len(sequence)))
    if endPosition > len(sequence):
        raise ValueError(("Ending positions must be less than or equal to the sequence "
                          "length. Found an ending position of {0} with a sequence "
                          "length of {1}.").format(endPosition, len(sequence)))
    if (endPosition - startPosition) < nMutBase:
        raise ValueError(("Fewer bases exist in the substring specified by the starting "
                          "and ending positions than need to be mutated. There are only "
                          "{0} currently, but {1} bases must be mutated at a "
                          "time").format(endPosition - startPosition, nMutBase))

    allAlts = []
    for _, ref in enumerate(sequence):
        alts = []
        for base in refSeq.BASES_ARR:
            if base == ref:
                continue
            alts.append(base)
        allAlts.append(alts)
    allMutations = []
    for indices in itertools.combinations(
            range(startPosition, endPosition), nMutBase):
        altsAtPos = []
        for i in indices:
            altsAtPos.append(allAlts[i])
        for mutations in itertools.product(*altsAtPos):
            allMutations.append(list(zip(indices, mutations)))
    return allMutations


def mutateSeqEnc(encoding, mutations, refSeq = Genome):
    """
    Introduce mutations to the given encoding of a sequence
    Transforms the encoding of a sequence to obtain.

    Parameters
    ----------
    encoding : numpy.ndarray
        An :math:`L \\times N` array (where :math:`L` is the sequence's
        length and :math:`N` is the size of the sequence type's
        alphabet) holding the one-hot encoding of the
        reference sequence.
    mutations : list(tuple)
        List of tuples of (`int`, `str`). Each tuple is the position to
        mutate and the base to which to mutate that position in the
        sequence.
    refSeq : class, optional
        Default is `uavarprior.data.sequences.Genome`. A reference sequence
        from which to retrieve smaller sequences..

    Returns
    -------
    numpy.ndarray
        An :math:`L \\times N` array holding the one-hot encoding of
        the mutated sequence.
    
    Note
    -------
    Modified from Selene's mutate_sequence function
    
    """
    mutEnc = np.copy(encoding)
    for (position, alt) in mutations:
        altIdx = refSeq.BASE_TO_INDEX[alt]
        mutEnc[position, :] = 0
        mutEnc[position, altIdx] = 1
    return mutEnc


def _mutationId(sequence, mutations):
    """
    Fomulate ID for in-silico generated mutations

    Parameters
    ----------
    sequence : str
        The input sequence to mutate.
    mutations : list(tuple)
        TODO

    Returns
    -------
    TODO
        TODO
    
    Note
    ------
    Modified from Selene's _ism_sample_id function
    """
    positions = []
    refs = []
    alts = []
    for (position, alt) in mutations:
        positions.append(str(position))
        refs.append(sequence[position])
        alts.append(alt)
    return (';'.join(positions), ';'.join(refs), ';'.join(alts))

