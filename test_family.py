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

from family import pdf
from family import trio_model as fm
from family import utilities as ut


class TestTree(unittest.TestCase):

    def setUp(self):
        self.muta_rate = 0.001
        self.nt_freq = ut.dc_alpha_parameters()
        # assume true if pass test_pop_sample
        self.parent_prob_mat = fm.pop_sample(self.muta_rate, self.nt_freq)
        
    # at population sample, events should sum to 1
    def test_pop_sample(self):
        parent_prob_mat = fm.pop_sample(self.muta_rate, self.nt_freq)
        proba = ut.sum_exp(parent_prob_mat)
        self.assertAlmostEqual(proba, 1)

    # at germline mutation, events should sum to 1
    # must condition on parent genotype layer
    def test_germ_muta(self):
        child_prob_mat = np.zeros((
            ut.GENOTYPE_COUNT,
            ut.GENOTYPE_COUNT,
            ut.GENOTYPE_COUNT
        ))

        for mother_gt, mom_idx in ut.GENOTYPE_INDEX.items():
            for father_gt, dad_idx in ut.GENOTYPE_INDEX.items():
                for child_gt, child_idx in ut.GENOTYPE_INDEX.items():
                    child_given_parent = fm.germ_muta(
                        child_gt,
                        mother_gt,
                        father_gt,
                        self.muta_rate
                    )
                    parent = self.parent_prob_mat[mom_idx, dad_idx]  # log
                    event = child_given_parent * np.exp(parent)
                    child_prob_mat[mom_idx, dad_idx, child_idx] = event
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
        nt_counts = ut.enum_nt_counts(2)  # genotypes always 2-allele
        proba_mat = fm.seq_error(self.nt_freq, nt_counts)
        proba = ut.sum_exp(proba_mat)
        self.assertAlmostEqual(proba, 1)

    def test_trio_prob(self):
        pass

if __name__ == '__main__':
    unittest.main()