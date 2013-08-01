from math import exp, pow, log
import numpy as np
import pdf
import utilities
import scipy.special as sp

def trio_prob(read_child, read_mom, read_dad,
              pop_muta_rate, pop_nt_freq,
              germ_muta_rate, soma_muta_rate,
              dc_nt_freq, dc_disp, dc_bias):
    """
    Return the probability of read data given a set of parameters

    An implementation of the trio model for a single site.

    This function implements the model by calling the functions
    written on the left of the following diagram. The function names label
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

    Input
    -----
    The input parameters are broken up into categories
    Read data
        read_child - a 4-element nt count list [#A, #C, #G, #T]
        read_mom   - a 4-element nt count list [#A, #C, #G, #T]
        read_dad   - a 4-element nt count list [#A, #C, #G, #T]
    Population parameters
        pop_muta_rate  - a scalar in [0, 1]
        pop_nt_freq    - a 4-element nt frequency list [%A, %C, %G, %T]
    Germline mutation parameters
        germ_muta_rate - a scalar in [0, 1]
    Somatic mutation parameters
        soma_muta_rate - a scalar in [0, 1]
    Sequencing error parameters
        dc_nt_freq  - a 4-element Dirichlet distribution parameter list
        dc_disp     - a dispersion parameter
        dc_bias     - a bias parameter

    Output
    ------
    prob    - a scalar probability value indicating the probability
              of the read data given the parameters

    Notes
    -----

    """
    prob = 0


    return prob

def seq_error(error_rate, priors_mat, read_counts):
    """
    Calculate the probability of sequencing error 

    Inputs
    ------
    error_rate          - the sequencing error rate as a float
    priors_mat          - 16 x 4 x 4 numpy array of the Dirichlet 
                          distribution priors for DCM corresponding to
                          16 genotype possibilities x 4 reference allele
                          possibilities
    read_counts         - a list of form [#A, #C, #T, #G] of nucleotide reads

    Output
    ------
    Returns 16 x 4 probability matrix in log base e space

    Assumes each chromosome is equally-likely to be sequenced

    The probability is drawn from a Dirichlet multinomial distribution:
    this is a point of divergence from the Cartwright et al. paper mentioned
    in the other functions
    """
    (n_genotypes, n_nucleotides, n_nucleotides) = priors_mat.shape
    alpha_mat = error_rate * priors_mat

    proba_mat = np.zeros((n_genotypes, n_nucleotides))
    for i in xrange(n_genotypes):
        for j in xrange(n_nucleotides):
            proba_mat[i, j] = pdf.dirichlet_multinomial(priors_mat[i, j, :],
                                                        read_counts) 

    return proba_mat 

def soma_muta(soma1, chrom1, muta_rate):
    """
    Calculate the probability of somatic mutation

    Terms refer to that of equation 5 on page 7 of Cartwright et al.: Family-
    Based Method for Capturing De Novo Mutations
    """
    proba = 0.0

    exp_term = np.exp(-4.0/3.0 * muta_rate)
    term1 = 0.25 - 0.25 * exp_term
    #term2 is indicator term

    #check if indicator function is true for each chromosome
    if soma1 == chrom1:
        ind_term_chrom1 = exp_term
    else:
        ind_term_chrom1 = 0

    #calculate the probability
    proba = term1 + ind_term_chrom1

    return proba

def germ_muta(child_chrom1, child_chrom2, mom_chrom1, mom_chrom2, dad_chrom1,
              dad_chrom2, muta_rate):
    """
    Determine the probability of germline mutations and parent chromosome 
    donation in the same step
    Supposes child_chrom1 is associated with the mother and child_chrom2 is 
    associated with the father
    """
    proba = 0.0
    
    exp_term = exp(-4.0/3.0 * muta_rate)
    homo_match = 0.25 + 0.75 * exp_term
    hetero_match = 0.25 + 0.25 * exp_term
    no_match = 0.25 - 0.25 * exp_term

    if mom_chrom1 == mom_chrom2:
        #then mom homozygous
        if child_chrom1 == mom_chrom1:
            term1 = homo_match
        else:
            term1 = no_match
    else:
        #then mom heterozygous
        if child_chrom1 == mom_chrom1 or child_chrom1 == mom_chrom2:
            term1 = hetero_match
        else:
            term1 = no_match

    if dad_chrom1 == dad_chrom2:
        #then dad homozygous
        if child_chrom2 == dad_chrom1:
            term2 = homo_match
        else:
            term2 = no_match
    else:
        #then dad heterozygous
        if child_chrom2 == dad_chrom1 or child_chrom2 == dad_chrom2:
            term2 = hetero_match
        else:
            term2 = no_match

    proba = term1 * term2
    
    return proba

def pop_sample(muta_rate, nt_freq):
    """
    Given a mutation rate parameter theta and a set of nucleotide
    appearance frequencies in the gene pool 
    (alpha_A, alpha_C, alpha_G, alpha_T),
    return a 16x16 probability matrix
    where the ij'th entry in the matrix is the probability that
    the mother has genotype i and the father has genotype j where
    i, j \in {AA, AC, AG, AT, 
              CA, CC, CG, CT,
              GA, GC, GG, GT,
              TA, TC, TG, TT}
    and probabilities are drawn from a Dirichlet multinomial distribution

    The 16 x 16 matrix is an order-relevant representation of
    the possible events in the sample space where the first
    dimension is one parent 2-allele genotype (at the nucleotide
    level a size 4 * 4 = 16 sample space) and the second dimension
    is the 2-allele genotype of another parent. For example

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

    The multinomial component of the model generates the nucleotide frequency
    parameter vector (alpha_A, alpha_C, alpha_G, alpha_T) based on the
    nucleotide count input data.

    The Dirichlet component of our models 
    uses this frequency parameter vector along with
    the 
        mutation rate, theta; 
        nucleotide frequences, [alpha_A, alpha_C, alpha_G, alpha_T]
        the genome nucleotide counts, [n_A, n_C, n_G, n_T]; and  

    and draws probabilities from the pdf
        \frac{\gamma(\theta)}{\gamma(\theta + N)} *
            \Pi_{i = A, C, G, T} \frac{\gamma(\alpha_i * \theta + n_i)}
                                      {\gamma(\alpha_i * \theta}

    We refer to the first term in the product as the constant_term 
    (because its value doesn't vary with the number of nucleotide counts) and
    the second term in the product as the product_term

    Values are calculated and returned in log base e space

    For example the genome mutation rate, theta, may be the small scalar
    quantity \theta = 0.00025, the frequency parameter
    vector (alpha_A, alpha_C, alpha_G, alpha_T) = (0.25, 0.25, 0.25, 0.25),
    the genome nucleotide counts (n_A, n_C, n_G, n_T) = (4, 0, 0, 0) for
    the event that both the mother and the father have genotype AA, 
    resulting in N = 4.

    Note: this model does not follow that of the Cartwright paper mentioned
    in other functions
    """
    # combine parameters for call to dirichlet multinomial
    muta_nt_freq = nt_freq
    for i in xrange(len(nt_freq)):
        muta_nt_freq[i] = nt_freq[i] * muta_rate

    # 16 x 16 lexicographical ordering of 2-allele genotypes
    # x 4 types of nucleotides
    genotype_count = utilities.two_parent_counts()
    (n_mother_geno, n_father_geno, n_nucleotides) = genotype_count.shape
    total_nucleotides = 4 # 2 parents x 2-allele genotype
    proba_mat = np.zeros((n_mother_geno, n_father_geno))
    for i in xrange(n_mother_geno):
        for j in xrange(n_father_geno):
            nt_count = genotype_count[i, j, :]
            log_proba = pdf.dirichlet_multinomial(muta_nt_freq, nt_count)
            proba_mat[i, j] = log_proba

    return proba_mat 

if __name__ == '__main__':
    error_rate = 0.001
    priors_mat = utilities.dc_alpha_parameters() 
    reads = [10, 10, 10, 10]
    print seq_error(error_rate, priors_mat, reads)
