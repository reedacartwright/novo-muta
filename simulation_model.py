import numpy as np

from family.trio_model import TrioModel
from family import utilities as ut


class SimulationModel(object):
    """
    A SimulationModel class used to validate TrioModel results using default
    parameter values. It generates a random sample and family pedigree.

    Attributes:
        trio_model: TrioModel object that contains all default parameters.
        exp_count: Coverage or the number of experiments.
        mom_gt: Mother's genotype.
        dad_gt: Father's genotype.
        child_gt: Child's genotype.
        child_germ_gt: Child's germline genotype.
        mom_soma_gt: Mother's somatic genotype.
        dad_soma_gt: Father's somatic genotype.
        child_soma_gt: Child's somatic genotype.
        mom_read: Read counts of the mother [#A, #C, #G, #T].
        dad_read: Read counts of the father [#A, #C, #G, #T].
        child_read: Read counts of the child [#A, #C, #G, #T].
        proba: A float representing the probability of mutation.
    """
    def __init__(self):
        """Generate by default a random sample with 50x coverage."""
        self.trio_model = TrioModel()
        self.exp_count = 50
        self._initialize()

    def _initialize(self):
        """Initialize all other parameters and set reads to the trio model."""
        self.mom_gt = np.random.choice(ut.GENOTYPES)
        self.dad_gt = np.random.choice(ut.GENOTYPES)
        self.child_gt = (self.mom_gt[np.random.randint(0, 2)] +
                         self.dad_gt[np.random.randint(0, 2)])
        self.child_germ_gt = self.muta(self.child_gt, is_soma=False)
        self.mom_soma_gt = self.muta(self.mom_gt)
        self.dad_soma_gt = self.muta(self.dad_gt)
        self.child_soma_gt = self.muta(self.child_germ_gt)
        self.mom_read = self.multinomial_sample(self.mom_soma_gt)
        self.dad_read = self.multinomial_sample(self.dad_soma_gt)
        self.child_read = self.multinomial_sample(self.child_soma_gt)
        self.trio_model.reads = [self.child_read, self.mom_read, self.dad_read]
        self.proba = self.trio_model.trio()

    def muta(self, gt, is_soma=True):
        """
        Mutate a genotype if random mutation rate is less than the germline or
        somatic mutation rates.

        Args:
            gt: The genotype to be mutated.
            is_soma: True to use somatic mutation rate, False to use germline
                mutation rate.

        Returns:
            The mutated genotype String.
        """
        muta_gt = ''
        muta_nts = {'A': ['C', 'G', 'T'],
                    'C': ['A', 'G', 'T'],
                    'G': ['A', 'C', 'T'],
                    'T': ['A', 'C', 'G']}
        rand = np.random.random_sample(2)  # 2 samples from [0, 1)
        if is_soma:
            muta_rate = self.trio_model.soma_muta_rate
        else:
            muta_rate = self.trio_model.germ_muta_rate
        for i, num in enumerate(rand):
            if num < muta_rate:
                muta_nt_pool = muta_nts.get(gt[i])
                muta_nt = np.random.choice(muta_nt_pool)
                muta_gt += muta_nt
            else:
                muta_gt += gt[i]
        return muta_gt

    def multinomial_sample(self, soma_gt):
        """
        Use an alpha based on the somatic genotype to select nucleotide
        frequencies and draw sequencing reads at a specified coverage (multinomial).

        Args:
            soma_gt: The somatic genotype to get the alpha.

        Returns:
            A read count [#A, #C, #G, #T].
        """
        alphas = ut.get_alphas(self.trio_model.seq_err_rate)
        soma_idx = ut.GENOTYPE_INDEX.get(soma_gt)
        alpha = alphas[soma_idx]
        rand_alpha = np.random.dirichlet(alpha)
        return np.random.multinomial(self.exp_count, rand_alpha)

    def print_all(self):
        """Print all details of one generated sample."""
        print('Mother genotype: %s' % self.mom_gt)
        print('Father genotype: %s' % self.dad_gt)
        print('Child genotype: %s' % self.child_gt)
        print('Child germline genotype: %s' % self.child_germ_gt)
        print('Mother somatic genotype: %s' % self.mom_soma_gt)
        print('Father somatic genotype: %s' % self.dad_soma_gt)
        print('Child somatic genotype: %s' % self.child_soma_gt)
        print('Mother read: ', end='')
        print(self.mom_read)
        print('Father read: ', end='')
        print(self.dad_read)
        print('Child read: ', end='')
        print(self.child_read)
        print('Probability of mutation: %s' % str(self.proba))

SimulationModel().print_all()