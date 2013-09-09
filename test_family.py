#!/usr/bin/env python
"""
P(T) probability of true mother genotype
P(S) probability of somatic mother genotype
P(R) probability of sequencing reads

P(T) requires no conditioning
P(S) = \sum_T P(T) * P(S|T)
P(R) = \sum_S P(S) * P(R|S)
"""
import math
import unittest

import numpy as np

from family import trio_model as fm
from family import utilities as ut


class TestTree(unittest.TestCase):

    def setUp(self):
        self.muta_rate = 0.001
        self.germ_muta_rate = 0.00000002
        self.nt_freq = ut.dc_alpha_parameters()
        # assume true if pass test_pop_sample
        self.parent_prob_mat = fm.pop_sample(self.muta_rate, self.nt_freq)
        self.nt_counts = ut.enum_nt_counts(2)  # genotypes always 2-allele
        self.false_pos = [[13, 4, 0, 0], [18, 2, 0, 0], [22, 0, 0, 0]]

        
    # at population sample, events should sum to 1
    def test_pop_sample(self):
        parent_prob_mat = fm.pop_sample(self.muta_rate, self.nt_freq)
        proba = ut.sum_exp(parent_prob_mat)
        self.assertAlmostEqual(proba, 1)

    # at germline mutation, events should sum to 1
    # must condition on parent genotype layer
    def test_germ_muta(self):
        child_prob_mat = fm.get_child_prob_mat(
            self.parent_prob_mat,
            self.muta_rate
        )
        proba = np.sum(child_prob_mat)
        self.assertAlmostEqual(proba, 1)

    # at somatic mutation, events should sum to 1
    # must condition on parent genotype layer
    def test_soma_muta(self):
        soma_given_geno = fm.get_soma_given_geno(self.muta_rate)
        # assign a pdf to the true genotype event space
        # based on the previous layer

        # collapse parent_prob_mat into a single parent
        geno = ut.sum_exp(self.parent_prob_mat, axis=0)
        proba = np.sum(geno)
        self.assertAlmostEqual(proba, 1)

        soma_and_geno = fm.join_soma(geno, soma_given_geno)
        soma_proba = np.sum(soma_and_geno)
        self.assertAlmostEqual(soma_proba, 1)

    def test_seq_error(self):
        proba_mat = fm.seq_error(0.005, self.nt_freq, self.nt_counts)
        proba = ut.sum_exp(proba_mat)
        self.assertAlmostEqual(proba, 1)

    def test_trio_prob(self):
        proba = fm.trio_prob(self.false_pos,
                             self.muta_rate, self.nt_freq,
                             self.germ_muta_rate, self.germ_muta_rate,
                             self.nt_freq, None, None)
        self.assertAlmostEqual(proba, 0.0015)

if __name__ == '__main__':
    unittest.main()