#!/usr/bin/env python
"""
Script to parse simulation input file, where the first column represents the
probability of mutation (as a float [0, 1]) and the second column represents if
the site contains a mutation (1 is True, 0 is False). Each site is placed on a
new line. This script creates 10 bins with probability cateogories in percentage
[0, 10), [10, 20), [20, 30), ..., [90, 100] and calculates the percentage of
the sites in each bin that contain a real mutation, meaning the value in
the has mutation column is 1. Assume there are 10 bins numbered 0-9. The digit
in the tenths place of the probability represents the number of the bin it
belongs to, except 100% will go in bin 9.
"""
import math
import sys

NUM_BINS = 10


fin = open(sys.argv[1])
lines = fin.readlines()
counts = [0] * NUM_BINS
total = [0] * NUM_BINS

for line in lines:
    values = line.strip('\n').split('\t')
    proba = float(values[0])
    has_muta = int(values[1])
    bin = int(min(math.floor(proba * NUM_BINS), NUM_BINS - 1))
    total[bin] += 1
    if has_muta == 1:
        counts[bin] += 1
fin.close()

for i in range(NUM_BINS):
    if total[i] > 0:
        has_muta_pc = counts[i] / total[i] * 100
        print('%.2f%% or %d/%d sites in bin %d contain a mutation.' %
            (has_muta_pc, counts[i], total[i], i))
    else:
        print('There are no sites in bin %d.' % i)