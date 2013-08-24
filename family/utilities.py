#!/usr/bin/env python
from math import pow
import numpy as np

# global constants for specifiying array size
# nt - nucleotide
# gt - genotype
NUCLEOTIDES = ['A', 'C', 'G', 'T']
NUCLEOTIDE_COUNT = len(NUCLEOTIDES)  # 4
GENOTYPES = []
for nt1 in NUCLEOTIDES:
    for nt2 in NUCLEOTIDES:
        GENOTYPES.append(nt1 + nt2)
GENOTYPE_COUNT = len(GENOTYPES)  # 16

# currently unused
# NT_INDEX = {}
# for i, nuc in enumerate(NUCLEOTIDES):
#     NT_INDEX.update({ nuc: i })
# N_NT = len(NT_INDEX)

# GENOTYPE_INDEX = {}
# for i, geno in enumerate(GENOTYPES):
#     GENOTYPE_INDEX.update({ geno: i })
# N_GT = len(GENOTYPE_INDEX)

# GENO_LEFT_EQUIV = {'AC':'CA', 'AG':'GA', 'AT':'TA',
#                    'CG':'GC', 'CT':'TC', 'GT':'TG'}
# GENO_RIGHT_EQUIV = {v:k for k, v in GENO_LEFT_EQUIV.items()}

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
        pow(NUCLEOTIDE_COUNT, size),
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

# if __name__ == '__main__':
#     print(enum_nt_counts(2))
#     print(enum_nt_counts(2).shape)
#     print(enum_nt_counts(3))
#     print(enum_nt_counts(3).shape)