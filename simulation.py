#!/usr/bin/env python
"""
Simulation program to validate TrioModel results using default parameter values.
"""
import numpy as np

from family.trio_model import TrioModel
from family import utilities as ut

mom_gt = np.random.choice(ut.GENOTYPES)
dad_gt = np.random.choice(ut.GENOTYPES)
child_nt1 = mom_gt[ np.random.randint(0, 2) ]  # 0 or 1
child_nt2 = dad_gt[ np.random.randint(0, 2) ]
child_gt = child_nt1 + child_nt2
print('Mother genotype: %s' % mom_gt)
print('Father genotype: %s' % dad_gt)
print('Child genotype: %s' % child_gt)

def muta(trio_model, gt, is_soma=True):
    """
    Mutate a genotype if random mutation rate is less than the germline or
    somatic mutation rates.

    Args:
        trio_model: TrioModel object that contains all default parameters
            including germline and somatic mutation rates.
        gt: The genotype to be mutated.
        is_geno: True to use somatic mutation rate, False to use germline
            mutation rate.

    Returns:
        The mutated genotype String.
    """
    muta_nts = {'A': ['C', 'G', 'T'],
                'C': ['A', 'G', 'T'],
                'G': ['A', 'C', 'T'],
                'T': ['A', 'C', 'G']}
    rand = np.random.random_sample(2)  # 2 samples from [0, 1)
    muta_rate = trio_model.soma_muta_rate if is_soma else trio_model.germ_muta_rate
    muta_gt = ''
    for i, num in enumerate(rand):
        if num < muta_rate:
            muta_nt_pool = muta_nts.get(gt[i])
            muta_nt = np.random.choice(muta_nt_pool)
            muta_gt += muta_nt
        else:
            muta_gt += gt[i]
    return muta_gt

sim_model = TrioModel()
child_germ_gt = muta(sim_model, child_gt, is_soma=False)
mom_soma_gt = muta(sim_model, mom_gt)
dad_soma_gt = muta(sim_model, dad_gt)
child_soma_gt = muta(sim_model, child_germ_gt)
print('Child germline genotype: %s' % child_germ_gt)
print('Mother somatic genotype: %s' % mom_soma_gt)
print('Father somatic genotype: %s' % dad_soma_gt)
print('Child somatic genotype: %s' % child_soma_gt)

def multinomial_sample(trio_model, soma_gt, exp_count):
    """
    Use an alpha based on the somatic genotype to select nucleotide
    frequencies and draw sequencing reads at a specified coverage (multinomial).

    Args:
        trio_model: TrioModel object that contains all default parameters
            including sequencing error rate.
        soma_gt: The somatic genotype to get the alpha.
        exp_count: Number of experiments.

    Returns:
        A read count [#A, #C, #G, #T].
    """
    alphas = ut.get_alphas(trio_model.seq_err_rate)
    soma_idx = ut.GENOTYPE_INDEX.get(soma_gt)
    alpha = alphas[soma_idx]
    rand_alpha = np.random.dirichlet(alpha)
    return np.random.multinomial(exp_count, rand_alpha)

exp_count = 50
mom_read = multinomial_sample(sim_model, mom_soma_gt, exp_count)
dad_read = multinomial_sample(sim_model, dad_soma_gt, exp_count)
child_read = multinomial_sample(sim_model, child_soma_gt, exp_count)
print('Mother read: ', end='')
print(mom_read)
print('Father read: ', end='')
print(dad_read)
print('Child read: ', end='')
print(child_read)

sim_model.reads = [child_read, mom_read, dad_read]
print('Probability of mutation: %s' % str(sim_model.trio()))