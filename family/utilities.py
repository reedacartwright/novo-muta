#!/usr/bin/env python
import itertools
import math

import numpy as np
from scipy import special as sp

# global constants for specifiying array size
NUCLEOTIDES = ['A', 'C', 'G', 'T']
NUCLEOTIDE_COUNT = len(NUCLEOTIDES)  # 4
NUCLEOTIDE_INDEX = {nt: i for i, nt in enumerate(NUCLEOTIDES)}

# is the order of genotypes relevant?
# use of genotype and index is consistent
# lexicographical ordered set of 2 nucleotide strings
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


def dirichlet_multinomial(alpha, n):
    """
    Calculate probability from the probability density function (pdf):

    \frac{\gamma(\theta)}{\gamma(\theta + N)} *
        \Pi_{i = A, C, G, T} \frac{\gamma(\alpha_i * \theta + n_i)}
                                  {\gamma(\alpha_i * \theta}

    We refer to the first term in the product as the constant_term,
    because its value doesn't vary with the number of nucleotide counts,
    and the second term in the product as the product_term.

    Args:
        alpha: A list of frequencies (doubles) for each category in the
            multinomial that sum to one.
        n: A list of samples (integers) for each category in the multinomial.

    Returns:
        A double equal to log_e(P) where P is the value calculated from the pdf.
    """
    N = sum(n)
    A = sum(alpha)
    constant_term = (sp.gammaln(A) - sp.gammaln(N + A))
    product_term = 0
    for i in range(len(n)):
        product_term += (sp.gammaln(alpha[i] + n[i]) - sp.gammaln(alpha[i]))
    return constant_term + product_term

def dc_alpha_parameters():
    """
    Generate Dirichlet multinomial alpha parameters
    alpha = (alpha_1, ..., alpha_K) for a K-category Dirichlet distribution
    (where K = 4 = #nt) that vary with each combination of parental
    genotype and reference nt.

    Returns:
        A 1 x 4 numpy array.
    """
    return np.array([0.25] * NUCLEOTIDE_COUNT)

def two_parent_counts():
    """
    Returns:
        A 16 x 16 x 4 numpy array of genotype counts where the (i, j) element
        of the array is the count of nucleotide k in the i'th mother genotype
        and the j'th father genotype with mother and father genotypes.
    """
    gt_count = np.zeros((
        GENOTYPE_COUNT,
        GENOTYPE_COUNT,
        NUCLEOTIDE_COUNT
    ))
    one_vec = np.ones((GENOTYPE_COUNT))
    for nt, nt_idx in NUCLEOTIDE_INDEX.items():
        for gt, gt_idx in GENOTYPE_INDEX.items():
            for base in gt:
                if base == nt:
                    gt_count[gt_idx, :, nt_idx] += one_vec  # mother
                    gt_count[:, gt_idx, nt_idx] += one_vec  # father

    return gt_count

def one_parent_counts():
    """
    Count the nucleotide frequencies for the 16 different 2-allele genotypes.

    Returns:
        A 16 x 4 numpy array where row/first dimension corresponds to the
        genotypes and column/second dimension is the frequency of each
        nucleotide.
    """
    counts = np.zeros(( GENOTYPE_COUNT, NUCLEOTIDE_COUNT ))
    for gt, gt_idx in GENOTYPE_INDEX.items():
        count_list = [0.0] * NUCLEOTIDE_COUNT
        for nt, nt_idx in NUCLEOTIDE_INDEX.items():
            for base in gt:
                if base == nt:
                    count_list[nt_idx] += 1
        counts[gt_idx, :] = count_list

    return counts

def enum_nt_counts(size):
    """
    Enumerate all nucleotide strings of a given size in lexicographic order.

    Args:
        size: The length of the nucleotide string.

    Returns:
        A 4^size x 4 numpy array of nucleotide counts associated with the
        strings.
    """
    nt_counts = np.zeros((
        math.pow(NUCLEOTIDE_COUNT, size),
        NUCLEOTIDE_COUNT
    ))
    first = np.identity(NUCLEOTIDE_COUNT)
    if size == 1:
        return first
    else:
        first_shape = first.shape
        second = enum_nt_counts(size - 1)  # recursive call
        second_shape = second.shape
        for j in range(second_shape[0]):
            for i in range(first_shape[0]):
                nt_counts[i+j*NUCLEOTIDE_COUNT, :] = (first[i, :] + second[j, :])
        return nt_counts