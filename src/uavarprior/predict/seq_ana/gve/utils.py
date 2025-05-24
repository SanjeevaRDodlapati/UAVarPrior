
"""
Note
-----
This is originated from Selene's _variant_effect_prediction.py
"""

from ....data.utils import formatChrom 


VCF_REQUIRED_COLS = ["#CHROM", "POS", "ID", "REF", "ALT"]


# TODO: Is this a general method that might belong in utils?
def read_vcf_file(input_path,
                  strand_index=None,
                  require_strand=False,
                  output_NAs_to_file=None,
                  seq_context=None,
                  reference_sequence=None):
    """
    Read the relevant columns for a variant call format (VCF) file to
    collect variants for variant effect prediction.

    Parameters
    ----------
    input_path : str
        Path to the VCF file.
    strand_index : int or None, optional
        Default is None. By default we assume the input sequence
        surrounding a variant is on the forward strand. If your
        model is strand-specific, you may specify a column number
        (0-based) in the VCF file that includes strand information. Please
        note that variant position, ref, and alt should still be specified
        for the forward strand and Selene will apply reverse complement
        to this variant.
    require_strand : bool, optional
        Default is False. Whether strand can be specified as '.'. If False,
        Selene accepts strand value to be '+', '-', or '.' and automatically
        treats '.' as '+'. If True, Selene skips any variant with strand '.'.
        This parameter assumes that `strand_index` has been set.
    output_NAs_to_file : str or None, optional
        Default is None. Only used if `reference_sequence` and `seq_context`
        are also not None. Specify a filepath to which invalid variants are
        written. Invalid = sequences that cannot be fetched, either because
        the exact chromosome cannot be found in the `reference_sequence` FASTA
        file or because the sequence retrieved based on the specified
        `seq_context` is out of bounds or overlapping with blacklist regions.
    seq_context : int or tuple(int, int) or None, optional
        Default is None. Only used if `reference_sequence` is not None.
        Specifies the sequence context in which the variant is centered.
        `seq_context` accepts a tuple of ints specifying the start and end
        radius surrounding the variant position or a single int if the
        start and end radius are the same length.
    reference_sequence : uavarprior.data.sequences.Genome or None, optional
        Default is None. Only used if `seq_context` is not None.
        The reference genome.

    Returns
    -------
    list(tuple)
        List of variants. Tuple = (chrom, position, id, ref, alt, strand)

    """
    variants = []
    na_rows = []
    withChrPrefix = True
    for chrom in reference_sequence.get_chrs():
        if not chrom.startswith("chr"):
            withChrPrefix = False
            break
    with open(input_path, 'r') as file_handle:
        lines = file_handle.readlines()
        index = 0
        for index, line in enumerate(lines):
            if '#' not in line:
                break
            if "#CHROM" in line:
                cols = line.strip().split('\t')
                if cols[:5] != VCF_REQUIRED_COLS:
                    raise ValueError(
                        "First 5 columns in file {0} were {1}. "
                        "Expected columns: {2}".format(
                            input_path, cols[:5], VCF_REQUIRED_COLS))
                index += 1
                break
        for line in lines[index:]:
            # print(line)
            cols = line.strip().split('\t')
            if len(cols) < 5:
                na_rows.append(line)
                continue
            chrom = str(cols[0])
            if 'CHR' == chrom[:3]:
                if withChrPrefix:
                    chrom = chrom.replace('CHR', 'chr')
                else:
                    chrom = chrom.replace('CHR', '')
            elif 'chr' in chrom and not withChrPrefix:
                chrom = formatChrom(chrom)
            elif "chr" not in chrom and withChrPrefix:
                chrom = "chr" + chrom

            if chrom == "chrMT" and \
                    chrom not in reference_sequence.get_chrs():
                chrom = "chrM"
            elif chrom == "MT" and \
                    chrom not in reference_sequence.get_chrs():
                chrom = "M"

            pos = int(cols[1])
            name = cols[2]
            ref = cols[3]
            if ref == '-':
                ref = ""
            alt = cols[4]
            strand = '+'
            if strand_index is not None:
                if require_strand and cols[strand_index] == '.':
                    na_rows.append(line)
                    continue
                elif cols[strand_index] == '-':
                    strand = '-'

            if reference_sequence and seq_context:
                if isinstance(seq_context, int):
                    seq_context = (seq_context, seq_context)
                lhs_radius, rhs_radius = seq_context
                start = pos + len(ref) // 2 - lhs_radius
                end = pos + len(ref) // 2 + rhs_radius
                if not reference_sequence.coords_in_bounds(chrom, start, end):
                    na_rows.append(line)
                    continue
            alt = alt.replace('.', ',')  # consider '.' a valid delimiter
            for a in alt.split(','):
                variants.append((chrom, pos, name, ref, a, strand))


    if reference_sequence and seq_context and output_NAs_to_file:
        with open(output_NAs_to_file, 'w') as file_handle:
            for na_row in na_rows:
                file_handle.write(na_row)
    return variants




