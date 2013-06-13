"""
This module groups together functions for calculating the probability
of mutations and errors (related to genome sequencing for a nuclear family)
that share a common probability distribution:
    P(i|j,u) = \frac{1}{4}(1 - e^{-4u/3}) + I(i = j)e^{-4u/3}
where i and j are nucleotides, u is a mutation or error rate, and I is
an indicator function
"""
import math.pow
def seq_error(read, soma, error_rate, p_chrom=None):
    """
    Calculate the probability of sequencing error 
    Inputs:
        read        - nucleotide read from sequencing as a string
        soma        - the somatic genotype that is sequenced as a list with
                      nucleotide elements for each chromosome
        error_rate  - the sequencing error rate as a float
        p_chrom     - the probability for selecting a chromosome from soma as 
                      a parallel list to soma; defaults for equal probabilities
    Returns the probability as a float
    """
    proba = 0.0
    e = 2.718
    
    #explicitly declare chromosomes
    chrom1 = soma[0]
    chrom2 = soma[1]

    #check for homozygous
    if chrom1 == chrom2:
        proba = (1/4 + 1/4 * math.pow(e,-4 * error_rate/3)
    else:
        if p_chrom is None:

        else:

    return proba

def soma_muta():

    return None

def germ_muta():

    return None
