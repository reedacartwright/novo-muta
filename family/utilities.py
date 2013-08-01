import numpy as np

def two_parent_counts():
    """
    Return the 16 x 16 x 4 numpy array of genotype counts where the
    ijk'th entry of the array is the count of nucleotide k
    in the i'th mother genotype and the j'th father genotype with
    mother and father genotypes in the lexicographical ordered set
    of 2 nucleotide strings
        {'AA', 'AC', 'AG', 'AT', 'CA', ...}
    """
    nucleotide_list = ['A', 'C', 'G', 'T']

    genotype_list = []
    for nucleotide1 in nucleotide_list:
        for nucleotide2 in nucleotide_list:
            genotype_list.append(nucleotide1 + nucleotide2)

    num_nucleotides = len(nucleotide_list)
    mother_genotypes = len(genotype_list)
    father_genotypes = len(genotype_list)

    genotype_count = np.zeros((mother_genotypes, father_genotypes,
                               num_nucleotides))
    one_vec = np.ones((len(genotype_list)))
    for nucleotide in xrange(num_nucleotides):
        for mother_geno in xrange(mother_genotypes):
            if genotype_list[mother_geno][0] == nucleotide_list[nucleotide]:
                genotype_count[mother_geno, :, nucleotide] += one_vec
            if genotype_list[mother_geno][1] == nucleotide_list[nucleotide]:
                genotype_count[mother_geno, :, nucleotide] += one_vec

        for father_geno in xrange(father_genotypes):
            if genotype_list[father_geno][0] == nucleotide_list[nucleotide]:
                genotype_count[:, father_geno, nucleotide] += one_vec
            if genotype_list[father_geno][1] == nucleotide_list[nucleotide]:
                genotype_count[:, father_geno, nucleotide] += one_vec

    return genotype_count

def one_parent_frequencies():
    """
    Count the nucleotide frequencies for the 16 different 2-allele genotypes

    Return a 16 x 4 np.array whose first dimension corresponds to the
    genotypes and whose second dimension is the frequency of each nucleotide
    """
    nucleotide_list = ['A', 'C', 'G', 'T']
    genotype_list = []
    for nucleotide1 in nucleotide_list:
        for nucleotide2 in nucleotide_list:
            genotype_list.append(nucleotide1 + nucleotide2)

    frequencies = np.zeros((len(genotype_list), len(nucleotide_list)))

    for gt in xrange(len(genotype_list)):
        count_list = [0.0, 0.0, 0.0, 0.0]
        for nt in xrange(len(nucleotide_list)):
            if genotype_list[gt][0] == nucleotide_list[nt]:
                count_list[nt] += 1
            if genotype_list[gt][1] == nucleotide_list[nt]:
                count_list[nt] += 1

        N = sum(count_list)
        freq_list = [0.0, 0.0, 0.0, 0.0]
        for i in xrange(len(freq_list)):
            freq_list[i] = count_list[i] / N

        frequencies[gt, :] = freq_list

    return frequencies

def dc_alpha_parameters():
    """
    Generate Dirichlet multinomial alpha parameters
    alpha = (alpha_1, ..., alpha_K) for a K-category Dirichlet distribution
    (where K = 4 = #nt) that vary with each combination of parental 
    genotype and reference nt
    """
    nt_index = {'A':0,
                'C':1,
                'G':2,
                'T':3}

    genotype_index = {'AA':0,
                      'AC':1,
                      'AG':2,
                      'AT':3,
                      'CA':4,
                      'CC':5,
                      'CG':6,
                      'CT':7,
                      'GA':8,
                      'GC':9,
                      'GG':10,
                      'GT':11,
                      'TA':12,
                      'TC':13,
                      'TG':14,
                      'TT':15}

    geno_left_equiv = {'AC':'CA',
                       'AG':'GA',
                       'AT':'TA',
                       'CG':'GC',
                       'CT':'TC',
                       'GT':'TG'}

    geno_right_equiv = {'CA':'AC',
                        'GA':'AG',
                        'TA':'AT',
                        'GC':'CG',
                        'TC':'CT',
                        'TG':'GT'}

    n_genotypes = len(genotype_index.keys())
    n_nt = len(nt_index.keys())

    # parental genotype, reference nt, alpha vector
    alpha_mat = np.zeros((n_genotypes, n_nt, n_nt))
    for i in xrange(n_genotypes):
        for j in xrange(n_nt):
            for k in xrange(n_nt):
                alpha_mat[i, j, k] = 0.25

    return alpha_mat

if __name__ == '__main__':
    print one_parent_frequencies()
