#!/usr/bin/env python
"""
P(T) probability of true mother genotype
P(S) probability of somatic mother genotype
P(R) probability of sequencing reads

P(T) requires no conditioning
P(S) = \sum_T P(T) * P(S|T)
P(R) = \sum_S P(S) * P(R|S)
"""
import unittest
import math
import numpy as np

from family import trio_model as fm
from family import utilities as ut
from family import pdf

class TestTree(unittest.TestCase):
    def setUp(self):
        self.muta_rate = 0.001
        self.nt_freq = [0.25] * ut.NUCLEOTIDE_COUNT
        # assume true if pass test_pop_sample
        self.parent_prob_mat = fm.pop_sample(self.muta_rate, self.nt_freq)
        
    # at the top (population sample) of the tree events should sum to 1
    def test_pop_sample(self):
        parent_prob_mat = fm.pop_sample(self.muta_rate, self.nt_freq)
        proba = np.sum( np.exp(parent_prob_mat) )
        self.assertAlmostEqual(proba, 1)

    # at the germline mutation level, we must condition on parent genotype layer
    # for events to sum to 1
    def test_germ_muta(self):
        child_prob_mat = np.zeros((
            ut.GENOTYPE_COUNT,
            ut.GENOTYPE_COUNT,
            ut.GENOTYPE_COUNT
        ))

        for mother_gt, mi in ut.GENOTYPE_INDEX.items():
            for father_gt, fi in ut.GENOTYPE_INDEX.items():
                for child_gt, ci in ut.GENOTYPE_INDEX.items():
                    child_given_parent = fm.germ_muta(child_gt, mother_gt,
                                                      father_gt, self.muta_rate)
                    parent = self.parent_prob_mat[mi, fi]
                    event = child_given_parent * np.exp(parent) # latter in log form
                    child_prob_mat[mi, fi, ci] = event
        proba = np.sum(child_prob_mat)
        self.assertAlmostEqual(proba, 1)

    # at the somatic mutation layer we again must condition on parent
    # genotype for events to sum to 1 
    def test_soma_muta(self):
        # first compute event space for somatic nucleotide given a genotype nucleotide
        # for a single chromosome
        prob_vec = np.zeros(( ut.NUCLEOTIDE_COUNT, ut.NUCLEOTIDE_COUNT ))
        for soma_nt in ut.NUCLEOTIDE_INDEX:
            i = ut.NUCLEOTIDE_INDEX[soma_nt]
            for geno_nt in ut.NUCLEOTIDE_INDEX:
                j = ut.NUCLEOTIDE_INDEX[geno_nt]
                prob_vec[i, j] = fm.soma_muta(soma_nt, geno_nt, self.muta_rate)

        # next combine event spaces for two chromosomes (independent of each other)
        # and call resulting 16x16 matrix soma_given_geno
        # first dimension lexicographical order of pairs of letters from 
        # nt alphabet for somatic genotypes
        # second dimension is that for true genotypes
        soma_given_geno = np.zeros(( ut.GENOTYPE_COUNT, ut.GENOTYPE_COUNT ))
        for chrom1 in ut.NUCLEOTIDE_INDEX:
            i = ut.NUCLEOTIDE_INDEX[chrom1]
            given_chrom1_vec = prob_vec[:, i]
            for chrom2 in ut.NUCLEOTIDE_INDEX:
                j = ut.NUCLEOTIDE_INDEX[chrom2]
                given_chrom2_vec = prob_vec[:, i]
                soma_muta_index = i * ut.NUCLEOTIDE_COUNT + j
                outer_prod = np.outer(given_chrom1_vec, given_chrom2_vec)
                outer_prod_flat = outer_prod.flatten()
                soma_given_geno[:, soma_muta_index] = outer_prod_flat

        # with the event space from the somatic mutation step calculated we can now
        # assign a pdf to the true genotype event space based on the previous layer

        # collapse parent prob mat into a single parent
        parent_prob_mat_exp = np.exp(self.parent_prob_mat)
        geno = np.sum(parent_prob_mat_exp, 0)
        # print(geno) # matrix
        proba = np.sum(geno)
        self.assertAlmostEqual(proba, 1)

        # and compute the joint probabilities
        soma_and_geno = np.zeros(( ut.GENOTYPE_COUNT, ut.GENOTYPE_COUNT ))
        for i in range(ut.GENOTYPE_COUNT):
            soma_and_geno[:, i] = geno[i] * soma_given_geno[:, i]

        # prob_soma = np.sum(soma_and_geno, 0) # matrix
        proba_soma = np.sum(soma_and_geno)
        self.assertAlmostEqual(proba_soma, 1)

        # finally, compute probabilities of sequencing error
        nt_string_size = 2
        nt_counts = ut.enum_nt_counts(nt_string_size)
        n_strings = int(math.pow( ut.NUCLEOTIDE_COUNT, nt_string_size ))
        alpha_params = np.zeros(( n_strings, ut.NUCLEOTIDE_COUNT ))
        prob_read_given_soma = np.zeros((n_strings))
        for i in range(n_strings):
            alpha_params[i] = [0.25] * ut.NUCLEOTIDE_COUNT
            log_prob = pdf.dirichlet_multinomial(alpha_params[i], nt_counts[i])
            prob_read_given_soma[i] = np.exp(log_prob)

        # print(prob_read_given_soma) # matrix
        proba = np.sum(prob_read_given_soma)
        self.assertAlmostEqual(proba, 1)

if __name__ == '__main__':
    unittest.main()