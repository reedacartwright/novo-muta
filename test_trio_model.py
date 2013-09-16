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
        # reads must be 3x4
        self.trio_model = TrioModel(
            reads=ut.enum_nt_counts(2),  # genotypes 2-allele
            pop_muta_rate=0.001,
            germ_muta_rate=0.00000002,
            soma_muta_rate=0.00000002,
            seq_err_rate=0.005,
            dm_disp=None,
            dm_bias=None
        )

    def test_pop_sample(self):
        parent_prob_mat = self.trio_model.pop_sample()
        proba = np.sum(parent_prob_mat)
        self.assertAlmostEqual(proba, 1)

    def test_germ_muta(self):
        child_prob_mat = self.trio_model.get_child_prob_mat()
        proba = np.sum(child_prob_mat)
        self.assertAlmostEqual(proba, 1)

    def test_soma_muta(self):
        soma_and_geno = self.trio_model.soma_and_geno()
        soma_proba = np.sum(soma_and_geno)
        self.assertAlmostEqual(soma_proba, 1)

    def test_seq_err(self):
        read_count = len(self.trio_model.reads)
        seq_prob_mat = np.zeros(( read_count, ut.GENOTYPE_COUNT ))
        for i in range(read_count):
            seq_prob_mat[i] = self.trio_model.seq_err(i)

        seq_prob_mat_scaled = ut.scale_to_log(
            seq_prob_mat,
            self.trio_model.max_elems
        )
        proba = np.sum(seq_prob_mat_scaled)
        self.assertAlmostEqual(proba, 16)  # 16 alpha freq

    def test_trio(self):
        proba = self.trio_model.trio()
        self.assertAlmostEqual(proba, 1)

if __name__ == '__main__':
    unittest.main()