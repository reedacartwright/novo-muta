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

import numpy as np

from family.trio_model import TrioModel
from family import utilities as ut


class TestTrioModel(unittest.TestCase):

    def setUp(self):
        self.trio_model = TrioModel(reads=ut.enum_nt_counts(2),  # genotypes always 2-allele
                                    pop_muta_rate=0.001,
                                    germ_muta_rate=0.00000002,
                                    soma_muta_rate=0.00000002,
                                    seq_err_rate=0.005,
                                    dm_disp=None,
                                    dm_bias=None)
        # assume true if pass test_pop_sample
        self.parent_prob_mat = self.trio_model.pop_sample(ut.ALPHAS[0])

    # at population sample, events should sum to ?
    def test_pop_sample(self):
        parent_prob_mat = self.trio_model.pop_sample(ut.ALPHAS[0])
        proba = np.sum(parent_prob_mat)
        # 4.00550122
        print(proba)
        pass

    # at germline mutation, events should sum to ?
    # must condition on parent genotype layer
    def test_germ_muta(self):
        child_prob_mat = self.trio_model.get_child_prob_mat(
            self.parent_prob_mat
        )
        proba = np.sum(child_prob_mat)
        # 4.00550122
        print(proba)
        pass

    # at somatic mutation, events should sum to ?
    # must condition on parent genotype layer
    def test_soma_muta(self):
        soma_given_geno = self.trio_model.get_soma_given_geno()
        geno = np.sum(self.parent_prob_mat, axis=0)
        proba = np.sum(geno)
        print(proba)
        # 4.014981

        soma_and_geno = TrioModel.join_soma(geno, soma_given_geno)
        soma_proba = np.sum(soma_and_geno)
        # 4.014981
        print(soma_proba)
        pass

    def test_seq_error(self):
        child_seq_prob = self.trio_model.seq_err(0)
        child_prob = np.sum(child_seq_prob)
        mom_seq_prob = self.trio_model.seq_err(1)
        mom_prob = np.sum(child_seq_prob)
        dad_seq_prob = self.trio_model.seq_err(2)
        dad_prob = np.sum(dad_seq_prob)

        # 4.00550122
        print(child_prob)
        print(mom_prob)
        print(dad_prob)
        pass

    def test_trio(self):
        proba = self.trio_model.trio()
        # 1
        print(proba)
        pass

if __name__ == '__main__':
    unittest.main()