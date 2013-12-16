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

    def test_pop_sample(self):
        parent_prob_mat = self.trio_model.pop_sample()
        proba = np.sum(parent_prob_mat)
        self.assertAlmostEqual(proba, 1)

    def test_germ_muta(self):
        proba = np.sum(self.trio_model.child_prob_mat)
        self.assertAlmostEqual(proba, 256)

    def test_soma_muta(self):
        proba = np.sum(self.trio_model.soma_prob_mat)
        self.assertAlmostEqual(proba, 16)

    def test_seq_err(self):
        seq_prob_mat = self.trio_model.seq_err_all()
        proba = np.sum(seq_prob_mat)
        self.assertAlmostEqual(proba, 3)

    def test_trio(self):
        proba = self.trio_model.trio()
        self.assertAlmostEqual(proba, 0)

if __name__ == '__main__':
    unittest.main()