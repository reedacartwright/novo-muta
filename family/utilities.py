"""
Utilities file containing useful contants and functions to support the TrioModel
including the Dirichlet multinomial.
"""
import itertools
import math

import numpy as np
from scipy import special as sp

# global constants for specifying array size
NUCLEOTIDES = ['A', 'C', 'G', 'T']
NUCLEOTIDE_COUNT = len(NUCLEOTIDES)  # 4
NUCLEOTIDE_INDEX = {nt: i for i, nt in enumerate(NUCLEOTIDES)}

# array of lexicographical ordered strings
GENOTYPES = ['%s%s' % pair
    for pair in itertools.product(NUCLEOTIDES, repeat=2)
]
GENOTYPE_COUNT = len(GENOTYPES)  # 16
GENOTYPE_INDEX = {gt: i for i, gt in enumerate(GENOTYPES)}


def get_alphas(rate):
    """
    Generate a 16 x 4 alpha frequencies matrix given the sequencing error rate.
    The order of the alpha frequencies is the same of that of GENOTYPES.

    Current values are placeholders until they have been estimated some time in
    Spring 2014.

    Args:
        rate: Float representing sequencing error rate.

    Returns:
        16 x 4 numpy array of Dirichlet multinomial alpha parameters
        alpha = (alpha_1, ..., alpha_K) for a K-category Dirichlet distribution
        (where K = 4 = NUCLEOTIDE_COUNT) that vary with each combination of
        parental genotype and reference nt.
    """
    return np.array([
        # A            C             G             T
        [1 - rate,     rate/3,       rate/3,       rate/3],
        [0.5 - rate/3, 0.5 - rate/3, rate/3,       rate/3],
        [0.5 - rate/3, rate/3,       0.5 - rate/3, rate/3],
        [0.5 - rate/3, rate/3,       rate/3,       0.5 - rate/3],

        [0.5 - rate/3, 0.5 - rate/3, rate/3,       rate/3],
        [rate/3,       1 - rate,     rate/3,       rate/3],
        [rate/3,       0.5 - rate/3, 0.5 - rate/3, rate/3],
        [rate/3,       0.5 - rate/3, rate/3,       0.5 - rate/3],
        
        [0.5 - rate/3, rate/3,       0.5 - rate/3, rate/3],
        [rate/3,       0.5 - rate/3, 0.5 - rate/3, rate/3],
        [rate/3,       rate/3,       1 - rate,     rate/3],
        [rate/3,       rate/3,       0.5 - rate/3, 0.5 - rate/3],

        [0.5 - rate/3, rate/3,       rate/3,       0.5 - rate/3],
        [rate/3,       0.5 - rate/3, rate/3,       0.5 - rate/3],
        [rate/3,       rate/3,       0.5 - rate/3, 0.5 - rate/3],
        [rate/3,       rate/3,       rate/3,       1 - rate]
    ])

def dirichlet_multinomial(alpha, n):
    """
    Calculate probability from the probability density function (pdf):

    \frac{\gamma(\theta)}{\gamma(\theta + N)} *
        \Pi_{i = A, C, G, T} \frac{\gamma(\alpha_i * \theta + n_i)}
                                  {\gamma(\alpha_i * \theta}

    Args:
        alpha: Array of floats representing frequencies for each category in the
            multinomial, and sum to one.
        n: Array of integers representing samples for each category in the
            multinomial.

    Returns:
        Double representing log_e(P) where P is the value calculated from the pdf.
    """
    N = sum(n)
    A = sum(alpha)
    constant_term = sp.gammaln(A) - sp.gammaln(N + A)
    product_term = 0
    for i, count in enumerate(n):
        product_term += sp.gammaln(alpha[i] + count) - sp.gammaln(alpha[i])
    return constant_term + product_term

def two_parent_counts():
    """
    Returns:
        16 x 16 x 4 numpy array of genotype counts where the (i, j) element of
        the array is the count of nucleotide k in the i'th mother genotype and
        the j'th father genotype with mother and father genotypes.
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
    Count the nucleotide frequencies for the 16 genotypes.

    Returns:
        16 x 4 numpy array where row/first dimension corresponds to the
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
        size: Integer representing length of the nucleotide string.

    Returns:
        4^size x 4 numpy array of nucleotide counts associated with the strings.
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
                nt_counts[i+j*NUCLEOTIDE_COUNT, :] = first[i, :] + second[j, :]
        return nt_counts

def normalspace(arr):
    """
    Rescale a numpy array in log space to normal space.

    Args:
        arr: numpy array.

    Returns:
        numpy array rescaled to normal space (the highest element is 1).
    """
    max_elem = np.amax(arr)
    return np.exp(arr-max_elem), max_elem

def logspace(arr, max_elem):
    """
    Scale a numpy array in normal space to log space knowing its max
    element.

    Currently used for testing purposes only.

    Args:
        arr: numpy array.
        max_elem: Greatest element in the array (stored when rescale_to_normal()
            is called).

    Returns:
        numpy array scaled to log space.
    """
    return np.log(arr) + max_elem

def logspace_all(arr, max_elems):
    """
    Scale a numpy array in normal space to log space.

    Currently used for testing purposes only.

    Args:
        arr: numpy multidimensional array.
        max_elems: Array containing the greatest element in each of the
            subarrays (stored when rescale_to_normal() is called).

    Returns:
        A numpy array scaled to log space.
    """
    for i in range(len(arr)):
        arr[i] = scale_to_log(arr[i], max_elems[i])
    return arr

def get_diag(arr):
    """
    Args:
        arr: numpy multidimensional array.

    Returns:
        Array with the major diagonal unchanged and all other elements replaced
        with 0.
    """
    return np.diag(np.diag(arr))