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
        self.seq_error_rate = 0.005
        self.nt_freq = ut.dc_alpha_parameters()
        self.nt_freqs = ut.dc_priors_mat()
        # assume true if pass test_pop_sample
        self.parent_prob_mat = fm.pop_sample(self.muta_rate, self.nt_freq)
        self.parent_prob_mat = ut.rescale_to_normal(self.parent_prob_mat)
        self.nt_counts = ut.enum_nt_counts(2)  # genotypes always 2-allele
        self.reads = [[13, 4, 0, 0], [18, 2, 0, 0], [22, 0, 0, 0]]

    # at population sample, events should sum to 1
    def test_pop_sample(self):
        parent_prob_mat = fm.pop_sample(self.muta_rate, self.nt_freq)
        parent_prob_mat = ut.rescale_to_normal(parent_prob_mat)
        proba = np.sum(parent_prob_mat)
        # 4.00550122
        pass

    # at germline mutation, events should sum to 1
    # must condition on parent genotype layer
    def test_germ_muta(self):
        child_prob_mat = fm.get_child_prob_mat(
            self.parent_prob_mat,
            self.germ_muta_rate
        )
        proba = np.sum(child_prob_mat)
        # 4.00550122
        pass

    # at somatic mutation, events should sum to 1
    # must condition on parent genotype layer
    def test_soma_muta(self):
        soma_given_geno = fm.get_soma_given_geno(self.germ_muta_rate)
        # assign a pdf to the true genotype event space
        # based on the previous layer

        # collapse parent_prob_mat into a single parent
        geno = np.sum(self.parent_prob_mat, axis=0)
        proba = np.sum(geno)
        # 4.00550122

        soma_and_geno = fm.join_soma(geno, soma_given_geno)
        soma_proba = np.sum(soma_and_geno)
        # 4.00550122
        pass

    def test_seq_error(self):
        proba_mat = fm.seq_error(self.nt_counts, self.nt_freqs,
                                 self.seq_error_rate)
        proba_mat = ut.rescale_to_normal(proba_mat)
        proba = np.sum(proba_mat)
        # 4.0149 * 16 genotypes = 64.238
        pass

    def test_trio_prob(self):
        proba = fm.trio_prob(self.reads,
                             self.muta_rate, self.nt_freq,
                             self.germ_muta_rate, self.germ_muta_rate,
                             self.seq_error_rate,
                             self.nt_freqs, None, None)
        pass

if __name__ == '__main__':
    unittest.main()