#!/usr/bin/env python
import math

import numpy as np

import pdf
import utilities as ut

def trio_prob(reads,
              pop_muta_rate, pop_nt_freq,
              germ_muta_rate, soma_muta_rate,
              dc_nt_freq, dc_disp, dc_bias):
    """
    Implement the trio model for a single site by calling the functions
    on the left of the following diagram. The function names label
    the arrow-denoted processes in the population model.

                          Population          Population
    pop_sample                |                   |
                              v                   v
                            Mother              Father
                            Zygotic             Zygotic
                            Diploid             Diploid
                            Genotype            Genotype
    germ_muta                 |   \             / |  
                              |    v           v  |
                              |      Daughter     |
                              |      Diploid      |
    soma_muta                 |         |         |
                              v         v         v
                            Mother   Daughter  Father
                            Somatic  Somatic   Somatic
                            Diploid  Diploid   Diploid
                            Genotype Genotype  Genotype
    seq_error                 |         |         |
                              v         v         v
                            Mother   Daughter  Father
                            Genotype Genotype  Genotype
                            Reads    Reads     Reads

    Args:
        reads: A list of 3 4-element nt count lists for child, mom, and dad
            [[#A, #C, #G, #T], [#A, #C, #G, #T], [#A, #C, #G, #T]].
        pop_muta_rate: A scalar in [0, 1].
        pop_nt_freq: A 4-element nt frequency list [%A, %C, %G, %T].
        germ_muta_rate: A scalar in [0, 1].
        soma_muta_rate: A scalar in [0, 1].
        dc_nt_freq: A 4-element Dirichlet distribution parameter list.
        dc_disp (unused): A dispersion parameter.
        dc_bias (unused): A bias parameter.

    Returns:
        A scalar probability of a mutation given read data and parameters.
    """
    # population sample mutation probability
    parent_prob_mat = pop_sample(pop_muta_rate, pop_nt_freq)  # 16x16
    pop_proba = ut.sum_exp(parent_prob_mat)

    # germline mutation probability
    child_prob_mat = get_child_prob_mat(parent_prob_mat, germ_muta_rate)
    germ_proba = np.sum(child_prob_mat)

    # somatic mutation probability
    soma_given_geno = get_soma_given_geno(soma_muta_rate)
    # assign a pdf to the true genotype event space
    # based on the previous layer

    # collapse parent_prob_mat into a single parent
    geno = ut.sum_exp(parent_prob_mat, axis=0)
    soma_proba = np.sum(geno)

    soma_and_geno = join_soma(geno, soma_given_geno)
    soma_and_geno_proba = np.sum(soma_and_geno)

    # sequencing error probability
    proba_mat = seq_error(0.005, dc_nt_freq, reads)
    seq_proba = ut.sum_exp(proba_mat)

    # calculate P(muta|data) = P(muta*data)/P(data)
    reads_given_params_proba = (pop_proba + germ_proba + soma_and_geno_proba
        + seq_proba)
    # TODO: implement formula for P(no muta,R|no muta)
    # http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3728889/
    no_muta_proba = 16.2222413646  
    no_muta_given_reads_proba = no_muta_proba/reads_given_params_proba
    return 1-no_muta_given_reads_proba

def seq_error(error_rate, priors_mat, reads):
    """
    Calculate the probability of sequencing error. Assume each chromosome is
    equally-likely to be sequenced.

    The probability is drawn from a Dirichlet multinomial distribution:
    This is a point of divergence from the Cartwright et al. paper mentioned
    in the other functions.

    Args:
        priors_mat: A 1 x 4 numpy array of the Dirichlet distribution
            priors.
        reads: A 2d array of nucleotide reads
            [[#A, #C, #T, #G], [#A, #C, #T, #G],].

    Returns:
        A probability matrix.
    """
    prob_read_given_soma = np.zeros((ut.GENOTYPE_COUNT))
    for i, read in enumerate(reads):
        prob_read_given_soma[i] = ut.dirichlet_multinomial(error_rate * priors_mat, read)

    return prob_read_given_soma

def soma_muta(soma, chrom, muta_rate):
    """
    Calculate the probability of somatic mutation.

    Terms refer to that of equation 5 on page 7 of Cartwright et al.: Family-
    Based Method for Capturing De Novo Mutations.

    Args:
        soma1: A nucleotide character.
        chrom1: Another nucleotide chracter to be compared.
        muta_rate: The mutation rate.

    Returns:
        The probability of somatic mutation.
    """
    exp_term = np.exp(-4.0/3.0 * muta_rate)
    term1 = 0.25 - 0.25 * exp_term
    # term2 is indicator term

    # check if indicator function is true for each chromosome
    ind_term_chrom1 = exp_term if soma == chrom else 0

    return term1 + ind_term_chrom1

def get_soma_vec(muta_rate):
    """
    Compute event space for somatic nucleotide given a genotype nucleotide
    for a single chromosome.

    Args:
        muta_rate: The mutation rate.

    Returns:
        A 4 x 4 probability vector.
    """
    prob_vec = np.zeros(( ut.NUCLEOTIDE_COUNT, ut.NUCLEOTIDE_COUNT ))
    for soma_nt, i in ut.NUCLEOTIDE_INDEX.items():
        for geno_nt, j in ut.NUCLEOTIDE_INDEX.items():
            prob_vec[i, j] = soma_muta(soma_nt, geno_nt, muta_rate)
    return prob_vec

def get_soma_given_geno(muta_rate):
    """
    Combine event spaces for two chromosomes (independent of each other).

    Args:
        muta_rate: The mutation rate.

    Returns:
        A 16 x 16 matrix where the first dimension is the lexicographically
        ordered pairs of letters from nt alphabet for somatic genotypes and
        second dimension is that for true genotypes.
    """
    prob_vec = get_soma_vec(muta_rate)
    soma_given_geno = np.zeros(( ut.GENOTYPE_COUNT, ut.GENOTYPE_COUNT ))
    for chrom1, i in ut.NUCLEOTIDE_INDEX.items():
        given_chrom1_vec = prob_vec[:, i]
        for chrom2, j in ut.NUCLEOTIDE_INDEX.items():
            given_chrom2_vec = prob_vec[:, i]
            soma_muta_index = i * ut.NUCLEOTIDE_COUNT + j
            outer_prod = np.outer(given_chrom1_vec, given_chrom2_vec)
            outer_prod_flat = outer_prod.flatten()
            soma_given_geno[:, soma_muta_index] = outer_prod_flat
    return soma_given_geno

def join_soma(geno, soma_given_geno):
    """
    Compute the joint probabilities.

    Args:
        geno: Probability array containing genotypes for one parent.
        soma_given_geno: A 16 x 16 matrix where the first dimension is the
            lexicographically ordered pairs of letters from nt alphabet for
            somatic genotypes and second dimension is that for true genotypes.

    Returns:
        A 16 x 16 probability matrix.
    """
    soma_and_geno = np.zeros(( ut.GENOTYPE_COUNT, ut.GENOTYPE_COUNT ))
    for i in range(ut.GENOTYPE_COUNT):
        soma_and_geno[:, i] = geno[i] * soma_given_geno[:, i]
    return soma_and_geno

def germ_muta(child_chrom, mom_chrom, dad_chrom, muta_rate):
    """
    Calculate the probability of germline mutation and parent chromosome 
    donation in the same step. Assume the first chromosome is associated with
    the mother and the second chromosome is associated with the father.

    Args:
        child_chrom: The 2-allele genotype string of the child.
        mom_chrom: The 2-allele genotype string of the mother.
        dad_chrom: The 2-allele genotype string of the father.
        muta_rate: The mutation rate.

    Returns:
        The probability of germline mutation.
    """
    exp_term = math.exp(-4.0/3.0 * muta_rate)
    homo_match = 0.25 + 0.75 * exp_term
    hetero_match = 0.25 + 0.25 * exp_term
    no_match = 0.25 - 0.25 * exp_term

    def get_term_match(parent_chrom, child_chrom_base):
        if child_chrom_base in parent_chrom:
            if parent_chrom[0] == parent_chrom[1]:
                return homo_match
            else:
                return hetero_match
        else:
            return no_match

    term1 = get_term_match(mom_chrom, child_chrom[0])
    term2 = get_term_match(dad_chrom, child_chrom[1])
    return term1 * term2

def get_child_prob_mat(parent_prob_mat, muta_rate):
    """
    Calculate the probability matrix for the offspring given the probability
    matrix of the parents and mutation rate.

    Args:
        parent_prob_mat: The 16 x 16 probability matrix of the parents in
        log e space (see pop_sample).
        muta_rate: The mutation rate.
    """
    child_prob_mat = np.zeros((
        ut.GENOTYPE_COUNT,
        ut.GENOTYPE_COUNT,
        ut.GENOTYPE_COUNT
    ))

    for mother_gt, mom_idx in ut.GENOTYPE_INDEX.items():
        for father_gt, dad_idx in ut.GENOTYPE_INDEX.items():
            for child_gt, child_idx in ut.GENOTYPE_INDEX.items():
                child_given_parent = germ_muta(
                    child_gt,
                    mother_gt,
                    father_gt,
                    muta_rate
                )
                parent = parent_prob_mat[mom_idx, dad_idx]  # log
                event = child_given_parent * np.exp(parent)
                child_prob_mat[mom_idx, dad_idx, child_idx] = event

    return child_prob_mat

def pop_sample(muta_rate, nt_freq):
    """
    The multinomial component of the model generates the nucleotide frequency
    parameter vector (alpha_A, alpha_C, alpha_G, alpha_T) based on the
    nucleotide count input data.

    Probabilities are drawn from a Dirichlet multinomial distribution. The
    Dirichlet component of our models uses this frequency parameter vector in
    addition to the mutation rate (theta), nucleotide frequencies
    [alpha_A, alpha_C, alpha_G, alpha_T], and genome nucleotide counts
    [n_A, n_C, n_G, n_T].

    For example: The genome mutation rate (theta) may be the small scalar
    quantity \theta = 0.00025, the frequency parameter vector
    (alpha_A, alpha_C, alpha_G, alpha_T) = (0.25, 0.25, 0.25, 0.25),
    the genome nucleotide counts (n_A, n_C, n_G, n_T) = (4, 0, 0, 0), for
    the event that both the mother and the father have genotype AA,
    resulting in N = 4.

    Note: This model does not follow that of the Cartwright paper mentioned
    in other functions.

    Args:
        muta_rate: A mutation rate parameter theta.
        nt_freq: A set of nucleotide appearance frequencies in the gene pool
            (alpha_A, alpha_C, alpha_G, alpha_T).

    Returns:
        A 16 x 16 probability matrix in log e space where the (i, j) element
        in the matrix is the probability that the mother has genotype i and
        the father has genotype j where i, j \in {AA, AC, AG, AT, 
                                                  CA, CC, CG, CT,
                                                  GA, GC, GG, GT,
                                                  TA, TC, TG, TT}.

        The matrix is an order-relevant representation of the possible events
        in the sample space where the first dimension is one parent 2-allele
        genotype (at the nucleotide level a size 4 * 4 = 16 sample space) and
        the second dimension is the 2-allele genotype of another parent. For
        example:

        P1/P2 | AA | AC | AG | AT | CA | CC | CG | CT | GA | ...
        -----
        AA   
        --
        AC
        --
        AG
        --
        AT
        --
        CA
        --
        .
        .
        .
    """
    # combine parameters for call to dirichlet multinomial
    muta_nt_freq = np.array([i * muta_rate for i in nt_freq])
    gt_count = ut.two_parent_counts()
    proba_mat = np.zeros(( ut.GENOTYPE_COUNT, ut.GENOTYPE_COUNT ))
    for i in range(ut.GENOTYPE_COUNT):
        for j in range(ut.GENOTYPE_COUNT):
            nt_count = gt_count[i, j, :]  # count per 2-allele genotype
            log_proba = ut.dirichlet_multinomial(muta_nt_freq, nt_count)
            proba_mat[i, j] = log_proba

    return proba_mat