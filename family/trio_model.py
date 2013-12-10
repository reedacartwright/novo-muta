import math

import numpy as np

from family import utilities as ut


class TrioModel(object):
    """
    A TrioModel object contains sequencing reads and parameters, which remain
    constant throughout the trio model. 

    Attributes:
        reads: 3 x 4 array containing read counts for child, mother, father
            [[#A, #C, #G, #T], [#A, #C, #G, #T], [#A, #C, #G, #T]].
        pop_muta_rate: Float representing the population mutation rate
            (theta).
        germ_muta_rate: Float representing germline mutation rate.
        soma_muta_rate: Float representing somatic mutation rate.
        seq_err_rate: Float representing sequencing error rate.
        dm_disp: Integer representing Dirichlet multinomial dispersion.
        dm_bias: Integer representing Dirichlet multinomial bias.
        max_elems: 1 x 3 array containing the greatest elements in the
            vectors calculated by seq_err() in order to rescale to log space.
        geno: 1 x 16 probability matrix of one parent dynamically derived
            from parent_prob_mat.
        parent_prob_mat: 16 x 16 probability matrix of the parents in log
            space (see pop_sample()).
        soma_and_geno_mat: 16 x 16 probability transition matrix of mutating
            to another genotype.
    """
    def __init__(self, reads=None,
                 pop_muta_rate=0.001, germ_muta_rate=0.00000002,
                 soma_muta_rate=0.00000002, seq_err_rate=0.005,
                 dm_disp=1000, dm_bias=None):
        """
        Initialize parameters. parent_prob_mat is dynamically generated by
        pop_sample() given the population mutation rate. geno is set when
        pop_sample() is called.
        """
        self.reads = reads
        self.pop_muta_rate = pop_muta_rate
        self.germ_muta_rate = germ_muta_rate
        self.soma_muta_rate = soma_muta_rate
        self.seq_err_rate = seq_err_rate
        self.dm_disp = dm_disp
        self.dm_bias = dm_bias
        self.max_elems = []
        self.geno = None
        self.parent_prob_mat = self.pop_sample()
        self.soma_and_geno_mat = self.soma_and_geno()

    def trio(self):
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
        
        See Cartwright et al.: Family-Based Method for Capturing De Novo
        Mutations.

        http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3728889/

        Returns:
            Float representing probability of mutation given read data and
            parameters.
        """     
        # multiply vectors by transition matrix
        seq_prob_mat = self.seq_err_all()  # 3x16
        child_prob = seq_prob_mat[0].dot(self.soma_and_geno_mat)  # 16
        mom_prob = seq_prob_mat[1].dot(self.soma_and_geno_mat)  # 16
        dad_prob = seq_prob_mat[2].dot(self.soma_and_geno_mat)  # 16

        # calculate denominator
        parent_prob = np.kron(dad_prob, mom_prob)  # 1x256
        child_prob_mat = self.get_child_prob_mat()  # 16x256
        child_germ_prob = child_prob.dot(child_prob_mat)  # 1x256
        dem_mat = np.multiply(child_germ_prob, parent_prob)
        dem_mat = np.multiply(dem_mat, self.parent_prob_mat)
        dem = np.sum(dem_mat)

        # calculate numerator
        soma_and_geno_diag = ut.get_diag(self.soma_and_geno_mat)  # 16x16
        child_prob_num = seq_prob_mat[0].dot(soma_and_geno_diag)  # 16
        mom_prob_num = seq_prob_mat[1].dot(soma_and_geno_diag)  # 16
        dad_prob_num = seq_prob_mat[2].dot(soma_and_geno_diag)  # 16

        child_prob_mat_num = self.get_child_prob_mat(no_muta=True)  # 16x256 
        child_germ_prob_num = child_prob_num.dot(child_prob_mat_num)  # 1x256
        parent_prob_num = np.kron(dad_prob_num, mom_prob_num) # 1x256
        num_mat = np.multiply(child_germ_prob_num, parent_prob_num)
        num_mat = np.multiply(num_mat, self.parent_prob_mat)
        num = np.sum(num_mat)

        no_muta_given_reads_proba = num/dem
        return 1-no_muta_given_reads_proba

    def seq_err(self, member):
        """
        Calculate the probability of sequencing error. Assume each chromosome
        is equally-likely to be sequenced.

        The probability is drawn from a Dirichlet multinomial distribution:
        This is a point of divergence from the Cartwright et al. paper
        mentioned in the other functions.

        When the Dirichlet multinomial is called, the max element is stored in
        max_elems, so that the scaling of the probability matrix can be
        manipulated later.

        Args:
            member: Integer representing index of the read counts for a
                family member in the trio model.

        Returns:
            1 x 16 probability vector that needs to be multiplied by a
            transition matrix.
        """
        # TODO: add bias when alpha freq are added
        alpha_mat = ut.get_alphas(self.seq_err_rate) * self.dm_disp

        prob_mat = np.zeros((ut.GENOTYPE_COUNT))
        for i, alpha in enumerate(alpha_mat):
            log_proba = ut.dirichlet_multinomial(alpha, self.reads[member])
            prob_mat[i] = log_proba

        prob_mat_rescaled, max_elem = ut.normalspace(prob_mat)
        self.max_elems.append(max_elem)

        return prob_mat_rescaled

    def seq_err_all(self):
        """
        Calculate the probability of sequencing error for all reads.

        Returns:
            3 x 16 probability matrix given that reads are a 3 x 4 array.
        """
        read_count = len(self.reads)
        seq_prob_mat = np.zeros(( read_count, ut.GENOTYPE_COUNT ))
        for i in range(read_count):
            seq_prob_mat[i] = self.seq_err(i)
        return seq_prob_mat

    def soma_muta(self, soma, chrom):
        """
        Calculate the probability of somatic mutation.

        Args:
            soma: String representing single nucleotide.
            chrom: String representing another nucleotide to be compared.

        Returns:
            Float representing probability of somatic mutation.
        """
        exp_term = np.exp(-4.0/3.0 * self.soma_muta_rate)
        term = 0.25 - 0.25 * exp_term

        # check if indicator function is true for each chromosome
        ind_term_chrom = exp_term if soma == chrom else 0

        return term + ind_term_chrom

    def soma_and_geno(self):
        """
        Compute event space for somatic nucleotide given a genotype nucleotide
        for a single chromosome. Combine event spaces for two chromosomes
        (independent of each other) and calculate somatic mutation probability
        matrix for a single parent.

        Returns:
            16 x 16 matrix where the first dimension is the lexicographically
            ordered pairs of letters from nt alphabet for somatic genotypes
            and second dimension is that for true genotypes.
        """
        prob_vec = np.zeros(( ut.NUCLEOTIDE_COUNT, ut.NUCLEOTIDE_COUNT ))
        for soma_nt, i in ut.NUCLEOTIDE_INDEX.items():
            for geno_nt, j in ut.NUCLEOTIDE_INDEX.items():
                prob_vec[i, j] = self.soma_muta(soma_nt, geno_nt)
        return np.kron(prob_vec, prob_vec)

    def germ_muta(self, child_nt, parent_gt, no_muta):
        """
        Calculate the probability of germline mutation and parent chromosome 
        donation in the same step. Assume the first chromosome is associated
        with the mother and the second chromosome is associated with the
        father.

        Args:
            child_nt: String representing child allele.
            parent_gt: 2-character string representing parent genotype.

        Returns:
            The probability of germline mutation.
        """
        exp_term = math.exp(-4.0/3.0 * self.germ_muta_rate)
        homo_match = 0.25 + 0.75 * exp_term
        hetero_match = 0.25 + 0.25 * exp_term
        no_match = 0.25 - 0.25 * exp_term

        if child_nt in parent_gt:
            if parent_gt[0] == parent_gt[1]:
                return homo_match
            else:
                return hetero_match if not no_muta else homo_match/2
        else:
            return no_match if not no_muta else 0

    def get_child_prob_mat(self, no_muta=False):
        """
        Calculate the probability matrix for the offspring given the
        probability matrix of the parents and mutation rate.

        Returns:
            16 x 256 probability matrix derived from the Kronecker product
            of a 4 x 16 probability matrix given one parent with itself.
        """
        child_prob_mat = np.zeros(( ut.NUCLEOTIDE_COUNT, ut.GENOTYPE_COUNT ))
        for nt, i in ut.NUCLEOTIDE_INDEX.items():
            for gt, j in ut.GENOTYPE_INDEX.items():
                child_prob_mat[i, j] = self.germ_muta(nt, gt, no_muta)
        return np.kron(child_prob_mat, child_prob_mat)

    def pop_sample(self):
        """
        The multinomial component of the model generates the nucleotide
        frequency parameter vector (alpha_A, alpha_C, alpha_G, alpha_T) based
        on the nucleotide count input data.

        Probabilities are drawn from a Dirichlet multinomial distribution.
        The Dirichlet component of our models uses this frequency parameter
        vector in addition to the mutation rate (theta), nucleotide
        frequencies [alpha_A, alpha_C, alpha_G, alpha_T], and genome
        nucleotide counts [n_A, n_C, n_G, n_T].

        For example: The genome mutation rate (theta) may be the small scalar
        quantity \theta = 0.00025, the frequency parameter vector
        (alpha_A, alpha_C, alpha_G, alpha_T) = (0.25, 0.25, 0.25, 0.25),
        the genome nucleotide counts (n_A, n_C, n_G, n_T) = (4, 0, 0, 0), for
        the event that both the mother and the father have genotype AA,
        resulting in N = 4.

        Note: This model does not follow that of the Cartwright paper
        mentioned in other functions.

        Set geno as the 1 x 16 probability matrix for a single parent.

        Returns:
            1 x 256 probability matrix in log e space where the (i, j)
            element in the matrix is the probability that the mother has
            genotype i and the father has genotype j where i, j in
            {AA, AC, AG, AT,
             CA, CC, CG, CT,
             GA, GC, GG, GT,
             TA, TC, TG, TT}.

            The matrix is an order-relevant representation of the possible
            events in the sample space that covers all possible parent
            genotype combinations. For example:

            [P(AAAA), P(AAAC), P(AAAG), P(AAAT), P(AACA), P(AACC), P(AACG)...]
        """
        # combine parameters for call to dirichlet multinomial
        muta_nt_freq = np.array([0.25 * self.pop_muta_rate for i in range(4)])
        gt_count = ut.two_parent_counts()
        prob_mat = np.zeros(( ut.GENOTYPE_COUNT, ut.GENOTYPE_COUNT ))
        for i in range(ut.GENOTYPE_COUNT):
            for j in range(ut.GENOTYPE_COUNT):
                nt_count = gt_count[i, j, :]  # count per 2-allele genotype
                log_proba = ut.dirichlet_multinomial(muta_nt_freq, nt_count)
                prob_mat[i, j] = np.exp(log_proba)

        self.geno = np.sum(prob_mat, axis=0)  # set one parent prob mat
        return prob_mat.flatten()