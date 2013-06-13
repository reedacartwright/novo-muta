"""
This module groups together functions for calculating the probability
of mutations and errors (related to genome sequencing for a nuclear family)
that share a common probability distribution:
    P(i|j,u) = \frac{1}{4}(1 - e^{-4u/3}) + I(i = j)e^{-4u/3}
where i and j are nucleotides, u is a mutation or error rate, and I is
an indicator function
"""
from math import exp, pow, log
import numpy as np
def seq_proba(reads, soma, error_rate):
    """
    Calculate the probability of sequencing error 
    Inputs:
        reads        - a dict of form {'A':#A,'C':#C,'T':#T,'G':#G} 
                      of nucleotide reads
                      each index \in {A,C,T,G}
        soma        - the somatic genotype that is sequenced as a list with
                      nucleotide elements for each chromosome e.g. ['A','T']
                      for AT-heterozygote
        error_rate  - the sequencing error rate as a float
    Returns the probability in log base e space
    Assumes each chromosome is equally-likely to be sequenced
    """
    proba = 0.0

    #determine the number of reads
    num_reads = 0
    for value in reads.itervalues():
        num_reads += value

    #explicitly declare chromosomes
    chrom1 = soma[0]
    chrom2 = soma[1]

    #store commonly used quantities in log e space
    exponential = exp(-4.0/3.0 * error_rate)
    homo_same = log(1.0/4.0 + 3.0/4.0 * exponential)
    hetero_same = log(1.0/4.0 + 1.0/4.0 * exponential)
    homo_diff = log(1.0/4.0 - 1.0/4.0 * exponential)
    hetero_diff = homo_diff

    if chrom1 == chrom2:
        #homozygous, check indicator function truth values
        num_same = reads[chrom1]
        num_diff = num_reads - num_same
        proba = num_same * homo_same + num_diff * homo_diff

    else:
        #heterozygous, again check indicator function
        num_same = reads[chrom1] + reads[chrom2]
        num_diff = num_reads - num_same
        proba = num_same * hetero_same + num_diff * hetero_diff

    return proba

def seq_error(reads, error_rate):
    """
    Return the read probabilities over all genotypes for a given
    sequencing error
    Notes:
        scaled refers to log-scaled
    """
    soma_list = [['A','A'], ['A','T'], ['A','C'], ['A','G'],
                 ['T','A'], ['T','T'], ['T','C'], ['T','G'],
                 ['C','A'], ['C','T'], ['C','C'], ['C','G'],
                 ['G','A'], ['G','T'], ['G','C'], ['G','G']]
    scaled_list = [0.0 for i in xrange(len(soma_list))]

    #first iteration of loop
    current_soma = soma_list[0]
    scaled_list[0] = seq_proba(reads, current_soma, error_rate)
    current_max = scaled_list[0]

    for index in xrange(1,len(soma_list)):
        soma = soma_list[index]
        proba = seq_proba(reads, soma, error_rate)
        scaled_list[index] = proba
        if proba > current_max:
            current_max = proba
            current_soma = soma

    return scaled_list, current_max

def soma_muta():

    return None

def germ_muta():

    return None

if __name__ == '__main__':
    reads = {'A':10, 'T':5, 'C':5, 'G':20}
    error_rate = 0.005
    proba_list, current_max = seq_error(reads, error_rate)
    print proba_list
    print current_max
