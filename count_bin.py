#!/usr/bin/env python
"""
Script to parse simulation input file. The first column represents the
probability of mutation. The second column represents if the site contains a
mutation either 1 or 0. Each site is placed on a new line.

This script creates bins with probability cateogories in percentage
[0, 10), [10, 20), [20, 30),...,[90, 100] and calculcates the percentage of
the sites in each bin that contain a real mutation, meaning the value in
the has mutation column is 1.
"""
import math
import sys

BIN_SIZE = 10


fin = open(sys.argv[1])
lines = fin.readlines()
total_counts = [0] * BIN_SIZE
counts = [0] * BIN_SIZE

for line in lines:
    values = line.strip('\n').split('\t')
    proba = float(values[0])
    has_muta = int(values[1])

    bin = int(min(math.floor(proba * 10), 9))
    total_counts[bin] += 1
    if has_muta == 1:
        counts[bin] += 1
fin.close()

for x in range(BIN_SIZE):
    if total_counts[x] > 0:
        has_muta_percent = counts[x] / total_counts[x] * 100
        print('%.2f%% or %d\\%d sites in bin %d contain a mutation.' %
            (has_muta_percent, counts[x], total_counts[x], x))
    else:
        print('There are no sites in bin %d.' % x)