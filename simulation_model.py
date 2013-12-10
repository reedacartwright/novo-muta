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
    def __init__(self, soma_muta_rate):
        """
        Generate a random sample and calculate probability of mutation with
        50x coverage.

        Somatic mutation rate is adjustable.
        """
        self.trio_model = TrioModel(soma_muta_rate=soma_muta_rate)
        self.cov = 50
        self.has_muta = False

    def muta(self, gt_idx):
        """
        Mutate genotype based on somatic transition matrix. Set has_muta to True
        if a mutation occurred, otherwise leave as False.

        Args:
            gt_idx: Integer representing index of the genotype to be mutated.

        Returns:
            Integer representing the index of the mutated genotype or the
            original genotype if no mutation occurred.
        """
        muta_gt_idx = np.random.choice(
            a=16,
            p=self.trio_model.soma_and_geno_mat[gt_idx]
        )
        if muta_gt_idx != gt_idx:
            self.has_muta = True
        return muta_gt_idx

    def dirichlet_multinomial_sample(self, soma_idx):
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
        alphas = ut.get_alphas(self.trio_model.seq_err_rate)
        rand_alpha = np.random.dirichlet(alphas[soma_idx])
        return np.random.multinomial(self.cov, rand_alpha)

    @classmethod
    def write_proba(cls, filename, exp_count, soma_muta_rate):
        """
        Generate exp_count samples and output their probabilities and whether
        that site contains a mutation (1 for True, 0 for False) to a file,
        each on a new line.
        
        Args:
            filename: String representing the name of the output file.
            exp_count: Integer representing the number of samples to generate.
            soma_muta_rate: Float representing somatic mutation rate.
        """
        sim_model = cls(soma_muta_rate=soma_muta_rate)
        parent_gt_idxs = np.random.choice(
            a=256,
            size=exp_count,
            p=sim_model.trio_model.parent_prob_mat
        )
        mom_gt_idxs = parent_gt_idxs % 16
        dad_gt_idxs = parent_gt_idxs // 16

        fout = open(filename, 'w')
        for m, d in itertools.zip_longest(mom_gt_idxs, dad_gt_idxs):
            child_gt_by_nt_idx = [np.random.choice(ut.GENOTYPE_NUM[m]),
                                  np.random.choice(ut.GENOTYPE_NUM[d])]
            child_gt_idx = ut.GENOTYPE_NUM.index(child_gt_by_nt_idx)
            child_germ_gt_idx = sim_model.muta(child_gt_idx)
            mom_soma_gt_idx = sim_model.muta(m)
            dad_soma_gt_idx = sim_model.muta(d)
            child_soma_gt_idx = sim_model.muta(child_germ_gt_idx)
            mom_read = sim_model.dirichlet_multinomial_sample(mom_soma_gt_idx)
            dad_read = sim_model.dirichlet_multinomial_sample(dad_soma_gt_idx)
            child_read = sim_model.dirichlet_multinomial_sample(child_soma_gt_idx)
            sim_model.trio_model.reads = [child_read, mom_read, dad_read]
            proba = str(sim_model.trio_model.trio())
            fout.write('%s\t%i\n' % (proba, sim_model.has_muta))
            sim_model.has_muta = False  # reset for next simulation
        fout.close()