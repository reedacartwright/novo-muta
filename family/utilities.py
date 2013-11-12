import itertools
import math

import numpy as np
from scipy import special as sp

# global constants for specifiying array size
NUCLEOTIDES = ['A', 'C', 'G', 'T']
NUCLEOTIDE_COUNT = len(NUCLEOTIDES)  # 4
NUCLEOTIDE_INDEX = {nt: i for i, nt in enumerate(NUCLEOTIDES)}

# lexicographical ordered set of 2 nucleotide strings
GENOTYPES = ['%s%s' % pair
    for pair in itertools.product(NUCLEOTIDES, repeat=2)
]
GENOTYPE_COUNT = len(GENOTYPES)  # 16
GENOTYPE_INDEX = {gt: i for i, gt in enumerate(GENOTYPES)}

# TODO: reduce genotypes from 16 to 10 by removing equivilants if efficiency
#    becomes an issue
# GENOTYPES = ['AA', 'AC', 'AG', 'AT', 'CC',
#              'CG', 'CT', 'GG', 'GT', 'TT']
GENOTYPE_LEFT_EQUIV = {
    'AC':'CA', 'AG':'GA', 'AT':'TA',
    'CG':'GC', 'CT':'TC', 'GT':'TG'
}
GENOTYPE_RIGHT_EQUIV = {v: k for k, v in GENOTYPE_LEFT_EQUIV.items()}


def get_alphas(rate):
    """
    Generate a 16 x 4 alpha matrix given the sequencing error rate. The order of
    the alpha frequencies is given in the same order of genotypes.

    Current values are placeholders for testing purposes.

    TODO: Replace with actual alpha frequencies when Rachel completes research
        or when parameters have been estimated.

    Args:
        rate: The sequencing error rate.

    Returns:
        A 16 x 4 numpy array of Dirichlet multinomial alpha parameters
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
    constant_term = sp.gammaln(A) - sp.gammaln(N + A)
    product_term = 0
    for i, count in enumerate(n):
        product_term += sp.gammaln(alpha[i] + count) - sp.gammaln(alpha[i])
    return constant_term + product_term

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
                nt_counts[i+j*NUCLEOTIDE_COUNT, :] = first[i, :] + second[j, :]
        return nt_counts

def normalspace(arr):
    """
    Rescale a numpy array in log space to normal space.

    Args:
        arr: A numpy array.

    Returns:
        A numpy array rescaled to normal space (the highest element is 1).
    """
    max_elem = np.amax(arr)
    return np.exp(arr-max_elem), max_elem

def logspace(arr, max_elem):
    """
    Scale a specific numpy array in normal space to log space knowing its max
    element.

    Currently used for testing purposes only.

    Args:
        arr: A numpy array.
        max_elem: The greatest element in the array (stored when
            rescale_to_normal is called).

    Returns:
        A numpy array scaled to log space.
    """
    return np.log(arr) + max_elem

def logspace_all(arr, max_elems):
    """
    Scale a numpy array in normal space to log space.

    Currently used for testing purposes only.

    Args:
        arr: A multidimensional numpy array.
        max_elems: A list of the greatest element in each of the subarrays
            (stored when rescale_to_normal is called).

    Returns:
        A numpy array scaled to log space.
    """
    for i in range(len(arr)):
        arr[i] = scale_to_log(arr[i], max_elems[i])
    return arr

def get_diag(arr):
    """
    Args:
        arr: A numpy multidimensional array.

    Returns:
        The array with the major diagonal constant and the rest are replaced
        with 0.
    """
    return np.diag(np.diag(arr))