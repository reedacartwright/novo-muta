import itertools

import numpy as np

from family.trio_model import TrioModel
from family import utilities as ut


class SimulationModel(object):
    """
    The SimulationModel class is used to validate TrioModel results using
    default parameter values described in the TrioModel class. It generates a
    random family pedigree based on population priors and calculates the
    probability of mutation using the generated sample (sequencing reads are
    drawn from the Dirichlet multinomial).

    Attributes:
        trio_model: TrioModel object that contains all default parameters.
        cov: Integer representing coverage or the number of experiments.
        has_muta: Boolean representing if this site contains a mutation.
    """
    def __init__(self, germ_muta_rate, soma_muta_rate):
        """
        Generate a random sample and calculate probability of mutation with
        50x coverage.

        Germline and somatic mutation rates are adjustable via command line.
        """
        if germ_muta_rate is not None and soma_muta_rate is not None:
            self.trio_model = TrioModel(germ_muta_rate=germ_muta_rate,
                                        soma_muta_rate=soma_muta_rate)
        elif germ_muta_rate is not None:
            self.trio_model = TrioModel(germ_muta_rate=germ_muta_rate)
        elif soma_muta_rate is not None:
            self.trio_model = TrioModel(soma_muta_rate=soma_muta_rate)
        else:
            self.trio_model = TrioModel()
        self.cov = 50
        self.has_muta = False

    def muta(self, gt_idx, is_soma=True):
        """
        Mutate genotype based on somatic transition matrix. Set has_muta to True
        if a mutation occurred, otherwise leave as False.

        Args:
            gt_idx: Integer representing index of the genotype to be mutated.
            is_soma: Set by default to True to use somatic mutation rate. Set to
                False to use germline mutation rate.

        Returns:
            Integer representing the index of the mutated genotype or the
            original genotype if no mutation occurred.
        """
        if is_soma:
            prob_mat = self.trio_model.soma_prob_mat[gt_idx]
        else:
            prob_mat = self.trio_model.child_prob_mat[:,gt_idx]
        muta_gt_idx = np.random.choice(a=ut.GENOTYPE_COUNT, p=prob_mat)
        if muta_gt_idx != gt_idx:
            self.has_muta = True
        return muta_gt_idx

    def dm_sample(self, soma_idx):
        """
        Use alpha frequencies based on the somatic genotype to select
        nucleotide frequencies and use these frequencies to draw sequencing
        reads at a specified coverage (Dirichlet multinomial).

        Args:
            soma_idx: Index of somatic genotype to get the appropriate alpha
            frequencies.

        Returns:
            Array containing read counts [#A, #C, #G, #T].
        """
        alpha_mat = (ut.get_alphas(self.trio_model.seq_err_rate) *
            self.trio_model.dm_disp)
        alpha = np.random.dirichlet(alpha_mat[soma_idx])
        return np.random.multinomial(self.cov, alpha)

    @classmethod
    def write_proba(cls, filename, exp_count, germ_muta_rate, soma_muta_rate):
        """
        Generate exp_count samples and output their probabilities and whether
        that site contains a mutation (1 for True, 0 for False) to a file,
        each on a new line.
        
        Args:
            filename: String representing the name of the output file.
            exp_count: Integer representing the number of samples to generate.
            germ_muta_rate: Float representing germline mutation rate.
            soma_muta_rate: Float representing somatic mutation rate.
        """
        sim_model = cls(
            germ_muta_rate=germ_muta_rate,
            soma_muta_rate=soma_muta_rate
        )
        parent_gt_idxs = np.random.choice(
            a=ut.GENOTYPE_COUNT * ut.GENOTYPE_COUNT,
            size=exp_count,
            p=sim_model.trio_model.parent_prob_mat
        )
        mom_gt_idxs = parent_gt_idxs % ut.GENOTYPE_COUNT
        dad_gt_idxs = parent_gt_idxs // ut.GENOTYPE_COUNT

        fout = open(filename, 'w')
        for m, d in itertools.zip_longest(mom_gt_idxs, dad_gt_idxs):
            child_gt_by_nt_idx = [
                np.random.choice(ut.GENOTYPE_NUM[m]),
                np.random.choice(ut.GENOTYPE_NUM[d])
            ]
            child_gt_idx = ut.GENOTYPE_NUM.index(child_gt_by_nt_idx)
            child_germ_gt_idx = sim_model.muta(
                m*ut.GENOTYPE_COUNT + d,
                is_soma=False
            )
            mom_soma_gt_idx = sim_model.muta(m)
            dad_soma_gt_idx = sim_model.muta(d)
            child_soma_gt_idx = sim_model.muta(child_germ_gt_idx)
            mom_read = sim_model.dm_sample(mom_soma_gt_idx)
            dad_read = sim_model.dm_sample(dad_soma_gt_idx)
            child_read = sim_model.dm_sample(child_soma_gt_idx)
            sim_model.trio_model.reads = [child_read, mom_read, dad_read]
            proba = str(sim_model.trio_model.trio())
            fout.write('%s\t%i\n' % (proba, sim_model.has_muta))
            sim_model.has_muta = False  # reset for next simulation
        fout.close()