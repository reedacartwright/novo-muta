#!/usr/bin/env python
import unittest

import numpy as np

from family.trio_model import TrioModel
from family import utilities as ut


class TestTrioModel(unittest.TestCase):
    def setUp(self):
        self.trio_model = TrioModel(
            reads=[[30, 0, 0, 0],
                   [30, 0, 0, 0],
                   [30, 0, 0, 0]]
        )
        self.generic_trio_model = TrioModel(reads=ut.enum_nt_counts(2))

    def test_pop_sample(self):
        parent_prob_mat = self.trio_model.pop_sample()
        proba = np.sum(parent_prob_mat)
        self.assertAlmostEqual(proba, 1)

    def test_germ_muta(self):
        child_prob_mat = self.trio_model.get_child_prob_mat()
        proba = np.sum(child_prob_mat)
        self.assertAlmostEqual(proba, 256)

    def test_soma_muta(self):
        soma_and_geno = self.trio_model.soma_and_geno()
        soma_proba = np.sum(soma_and_geno)
        self.assertAlmostEqual(soma_proba, 16)

    # use generic nucleotide set
    def test_seq_err(self):
        seq_prob_mat = self.generic_trio_model.seq_err_all()
        seq_prob_mat_scaled = ut.scale_to_log_all(
            seq_prob_mat,
            self.generic_trio_model.max_elems
        )
        proba = np.sum(seq_prob_mat_scaled)
        self.assertAlmostEqual(proba, 16)  # 16 alpha freq

    # use actual 3 x 4 read data
    def test_trio(self):
        proba = self.trio_model.trio()
        self.assertAlmostEqual(proba, 0)

if __name__ == '__main__':
    unittest.main()