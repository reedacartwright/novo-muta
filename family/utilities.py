#!/usr/bin/env python
import math
import itertools
import numpy as np

# global constants for specifiying array size
# nt - nucleotide
# gt - genotype
NUCLEOTIDES = ['A', 'C', 'G', 'T']
NUCLEOTIDE_COUNT = len(NUCLEOTIDES)  # 4
NUCLEOTIDE_INDEX = {nt: i for i, nt in enumerate(NUCLEOTIDES)}

# order of genotypes relevant? use of genotype and index is consistent
GENOTYPES = ['%s%s' % pair
    for pair in itertools.product(NUCLEOTIDES, repeat=2)
]
GENOTYPE_COUNT = len(GENOTYPES)  # 16
GENOTYPE_INDEX = {gt: i for i, gt in enumerate(GENOTYPES)}

GENOTYPE_LEFT_EQUIV = {
    'AC':'CA', 'AG':'GA', 'AT':'TA',
    'CG':'GC', 'CT':'TC', 'GT':'TG'
}
GENOTYPE_RIGHT_EQUIV = {v: k for k, v in GENOTYPE_LEFT_EQUIV.items()}

def two_parent_counts():
    """
    Return the 16 x 16 x 4 numpy array of genotype counts where the
    ijk'th entry of the array is the count of nucleotide k
    in the i'th mother genotype and the j'th father genotype with
    mother and father genotypes in the lexicographical ordered set
    of 2 nucleotide strings
        {'AA', 'AC', 'AG', 'AT', 'CA', ...}
    """
    gt_count = np.zeros((
        GENOTYPE_COUNT,
        GENOTYPE_COUNT,
        NUCLEOTIDE_COUNT
    ))
    one_vec = np.ones((GENOTYPE_COUNT))
    for nt in range(NUCLEOTIDE_COUNT):
        # mother genotype
        for mother_gt in range(GENOTYPE_COUNT):
            if GENOTYPES[mother_gt][0] == NUCLEOTIDES[nt]:
                gt_count[mother_gt, :, nt] += one_vec
            if GENOTYPES[mother_gt][1] == NUCLEOTIDES[nt]:
                gt_count[mother_gt, :, nt] += one_vec

        # father genotype
        for father_gt in range(GENOTYPE_COUNT):
            if GENOTYPES[father_gt][0] == NUCLEOTIDES[nt]:
                gt_count[:, father_gt, nt] += one_vec
            if GENOTYPES[father_gt][1] == NUCLEOTIDES[nt]:
                gt_count[:, father_gt, nt] += one_vec

    return gt_count

def one_parent_counts():
    """
    Count the nucleotide frequencies for the 16 different 2-allele genotypes

    Return a 16 x 4 np.array whose first dimension corresponds to the
    genotypes and whose second dimension is the frequency of each nucleotide
    """
    counts = np.zeros(( GENOTYPE_COUNT, NUCLEOTIDE_COUNT ))
    for gt in range(GENOTYPE_COUNT):
        count_list = [0.0] * NUCLEOTIDE_COUNT
        for nt in range(NUCLEOTIDE_COUNT):
            if GENOTYPES[gt][0] == NUCLEOTIDES[nt]:
                count_list[nt] += 1
            if GENOTYPES[gt][1] == NUCLEOTIDES[nt]:
                count_list[nt] += 1

        counts[gt, :] = count_list

    return counts

def enum_nt_counts(size):
    """
    Enumerate all nucleotide strings of a given size in lexicographic order
    and return a 4^size x 4 numpy array of nucleotide counts associated
    with the strings
    """
    nt_counts = np.zeros((
        math.pow(NUCLEOTIDE_COUNT, size),
        NUCLEOTIDE_COUNT
    ))
    if size == 1:
        return np.identity(NUCLEOTIDE_COUNT)
    else:
        first = np.identity(NUCLEOTIDE_COUNT)
        first_shape = (NUCLEOTIDE_COUNT, NUCLEOTIDE_COUNT)
        second = enum_nt_counts(size - 1)
        second_shape = second.shape
        for j in range(second_shape[0]):
            for i in range(first_shape[0]):
                nt_counts[i+j*NUCLEOTIDE_COUNT, :] = (first[i, :] + second[j, :])
        return nt_counts

def dc_alpha_parameters():
    """
    Generate Dirichlet multinomial alpha parameters
    alpha = (alpha_1, ..., alpha_K) for a K-category Dirichlet distribution
    (where K = 4 = #nt) that vary with each combination of parental 
    genotype and reference nt
    """
    # parental genotype, reference nt, alpha vector
    alpha_mat = np.zeros((
        GENOTYPE_COUNT,
        NUCLEOTIDE_COUNT,
        NUCLEOTIDE_COUNT
    ))
    for i in range(GENOTYPE_COUNT):
        for j in range(NUCLEOTIDE_COUNT):
            for k in range(NUCLEOTIDE_COUNT):
                alpha_mat[i, j, k] = 0.25

    return alpha_mat