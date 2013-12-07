import numpy as np

from family.trio_model import TrioModel
from family import utilities as ut


class SimulationModel(object):
    """
    The SimulationModel class is used to validate TrioModel results using
    default parameter values described in the TrioModel class. It generates a
    random family pedigree and calcuates the probability of mutation using the
    generated sample.

    The simulation will generate a random genotype for each parent and then
    picks one allele from each parent to become the child genotype. It mutates
    each allele in the child germline genotype if a random number drawn is less
    than the germline mutation rate. This step is repeated for somatic
    mutatation. Alpha frequencies are drawn based on the somatic genotypes to
    select nucleotide frequencies using the Dirichlet. These frequencies are
    used to draw sequencing reads at a specified coverage.

    The simulation can output the following, for example:

    Mother genotype: GC
    Father genotype: AA
    Child genotype: GA
    Child germline genotype: GA
    Mother somatic genotype: GC
    Father somatic genotype: AA
    Child somatic genotype: GA
    Mother read: 47C 3G
    Father read: 50A
    Child read: 8A 42G
    Probability of mutation: <This number will vary depending on parameters.>

    Attributes:
        trio_model: TrioModel object that contains all default parameters.
        cov: Integer representing coverage or the number of experiments.
        has_muta: Boolean representing if this site contains a mutation.
        mom_gt: 2-character string representing mother genotype.
        dad_gt: 2-character string representing father genotype.
        child_gt: 2-character string representing child genotype.
        child_germ_gt: 2-character string representing child germline genotype.
        mom_soma_gt: 2-character string representing mother somatic genotype.
        dad_soma_gt: 2-character string representing father somatic genotype.
        child_soma_gt: 2-character string representing child somatic genotype.
        mom_read: Array containing read counts of the mother [#A, #C, #G, #T].
        dad_read: Array containing read counts of the father [#A, #C, #G, #T].
        child_read: Array containing read counts of the child [#A, #C, #G, #T].
        proba: String representing the probability of mutation.
    """
    def __init__(self):
        """
        Generate a random sample and calculate probability of mutation with
        50x coverage.
        """
        self.trio_model = TrioModel()
        self.cov = 50
        self.has_muta = False
        self._initialize()

    def _initialize(self):
        """
        Initialize all other parameters not previously initialized, generate
        and set reads to the TrioModel object, and calculate the probability of
        mutation.
        """
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
        self.proba = str(self.trio_model.trio())

    def muta(self, gt, is_soma=True):
        """
        Mutate each allele in a genotype if a random number drawn [0, 1) is less
        than the germline or somatic mutation rate. Set has_muta to True if a
        mutation occurred, otherwise leave as False.

        Args:
            gt: 2-character string representing genotype to be mutated.
            is_soma: Set by default to True to use somatic mutation rate. Set to
                False to use germline mutation rate.

        Returns:
            2-character string representing the mutated genotype or the original
            genotype if no mutation occurred.
        """
        if is_soma:
            muta_rate = self.trio_model.soma_muta_rate
        else:
            muta_rate = self.trio_model.germ_muta_rate
        muta_gt = ''
        muta_nts = {'A': ['C', 'G', 'T'],
                    'C': ['A', 'G', 'T'],
                    'G': ['A', 'C', 'T'],
                    'T': ['A', 'C', 'G']}
        rand = np.random.random_sample(2)
        for allele, num in enumerate(rand):
            if num < muta_rate:
                muta_nt_pool = muta_nts.get(gt[allele])
                muta_nt = np.random.choice(muta_nt_pool)
                muta_gt += muta_nt
                self.has_muta = True
            else:
                muta_gt += gt[allele]
        return muta_gt

    def multinomial_sample(self, soma_gt):
        """
        Use alpha frequencies based on the somatic genotype to select
        nucleotide frequencies and use these frequencies to draw sequencing
        reads at a specified coverage (Dirichlet multinomial).

        Args:
            soma_gt: Somatic genotype to get the appropriate alpha frequencies.

        Returns:
            Array containing read counts [#A, #C, #G, #T].
        """
        alphas = ut.get_alphas(self.trio_model.seq_err_rate)
        soma_idx = ut.GENOTYPE_INDEX.get(soma_gt)
        alpha = alphas[soma_idx]
        rand_alpha = np.random.dirichlet(alpha)
        return np.random.multinomial(self.cov, rand_alpha)

    def print_all(self):
        """Print all details of sample including the probability of mutation."""
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
        print('Probability of mutation: %s' % self.proba)
        print('Has mutation: %s' % self.has_muta)

    @classmethod
    def write_proba(cls, filename, exp_count):
        """
        Generate exp_count samples and output their probabilities to a file,
        each on a new line.
        
        Args:
            filename: String representing the name of the output file.
            exp_count: Integer representing the number of samples to generate.
        """
        fout = open(filename, 'w')
        for x in range(exp_count):
            sim_model = cls()
            fout.write('%s\t%s\n' % (sim_model.proba, sim_model.has_muta))
        fout.close()