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
        self.soma_muta_rate = 0.00000002
        self.seq_error_rate = 0.005
        # assume true if pass test_pop_sample
        self.parent_prob_mat = fm.pop_sample(self.muta_rate, ut.ALPHAS[0])
        self.nt_counts = ut.enum_nt_counts(2)  # genotypes always 2-allele
        self.reads = [[13, 4, 0, 0], [18, 2, 0, 0], [22, 0, 0, 0]]

    # at population sample, events should sum to ?
    def test_pop_sample(self):
        parent_prob_mat = fm.pop_sample(self.muta_rate, ut.ALPHAS[0])
        proba = np.sum(parent_prob_mat)
        # 4.00550122
        print(proba)
        pass

    # at germline mutation, events should sum to ?
    # must condition on parent genotype layer
    def test_germ_muta(self):
        child_prob_mat = fm.get_child_prob_mat(
            self.parent_prob_mat,
            self.germ_muta_rate
        )
        proba = np.sum(child_prob_mat)
        # 4.00550122
        print(proba)
        pass

    # at somatic mutation, events should sum to ?
    # must condition on parent genotype layer
    def test_soma_muta(self):
        soma_given_geno = fm.get_soma_given_geno(self.germ_muta_rate)
        geno = np.sum(self.parent_prob_mat, axis=0)
        proba = np.sum(geno)
        # 4.00550122

        soma_and_geno = fm.join_soma(geno, soma_given_geno)
        soma_proba = np.sum(soma_and_geno)
        # 4.00550122
        print(soma_proba)
        pass

    def test_seq_error(self):
        proba_mat = np.array([
            fm.seq_error(nt_count, self.seq_error_rate)
            for nt_count in self.nt_counts
        ])
        proba = np.sum(proba_mat)
        # 256
        print(proba)
        pass

    def test_trio_prob(self):
        proba = fm.trio_prob(self.nt_counts,
                             self.muta_rate, ut.ALPHAS[0],
                             self.germ_muta_rate, self.soma_muta_rate,
                             self.seq_error_rate, dm_disp=None, dm_bias=None)
        # 1
        print(proba)
        pass

if __name__ == '__main__':
    unittest.main()