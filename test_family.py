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
        # assume true if pass test_pop_sample
        self.parent_prob_mat = fm.pop_sample(ut.ALPHAS[0])
        self.nt_counts = ut.enum_nt_counts(2)  # genotypes always 2-allele
        
    # at population sample, events should sum to ?
    def test_pop_sample(self):
        parent_prob_mat = fm.pop_sample(ut.ALPHAS[0])
        proba = np.sum(parent_prob_mat)
        # 4.00550122
        print(proba)
        pass

    # at germline mutation, events should sum to ?
    # must condition on parent genotype layer
    def test_germ_muta(self):
        child_prob_mat = fm.get_child_prob_mat(self.parent_prob_mat)
        proba = np.sum(child_prob_mat)
        # 4.00550122
        print(proba)
        pass

    # at somatic mutation, events should sum to ?
    # must condition on parent genotype layer
    def test_soma_muta(self):
        soma_given_geno = fm.get_soma_given_geno()
        geno = np.sum(self.parent_prob_mat, axis=0)
        proba = np.sum(geno)
        # 4.00550122

        soma_and_geno = fm.join_soma(geno, soma_given_geno)
        soma_proba = np.sum(soma_and_geno)
        # 4.00550122
        print(soma_proba)
        pass

    def test_seq_error(self):
        seq_prob_vecs = np.array([
            fm.seq_error(nt_count)
            for nt_count in self.nt_counts
        ])
        seq_prob_mat = []
        for vec in seq_prob_vecs:
            seq_prob_mat.append(vec * ut.TRANS_MAT)
        proba = np.sum(seq_prob_mat)
        # 16
        print(proba)
        pass

    def test_trio(self):
        proba = fm.trio(self.nt_counts, ut.ALPHAS[0])
        # 1
        print(proba)
        pass

if __name__ == '__main__':
    unittest.main()