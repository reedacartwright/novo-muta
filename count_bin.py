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
    if proba >= 0.9:
        total_counts[9] += 1
        if has_muta == 1:
            counts[9] += 1
    elif proba >= 0.8 and proba < 0.9:
        total_counts[8] += 1
        if has_muta == 1:
            counts[8] += 1
    elif proba >= 0.7 and proba < 0.8:
        total_counts[7] += 1
        if has_muta == 1:
            counts[7] += 1
    elif proba >= 0.6 and proba < 0.7:
        total_counts[6] += 1
        if has_muta == 1:
            counts[6] += 1
    elif proba >= 0.5 and proba < 0.6:
        total_counts[5] += 1
        if has_muta == 1:
            counts[5] += 1
    elif proba >= 0.4 and proba < 0.5:
        total_counts[4] += 1
        if has_muta == 1:
            counts[4] += 1
    elif proba >= 0.3 and proba < 0.4:
        total_counts[3] += 1
        if has_muta == 1:
            counts[3] += 1
    elif proba >= 0.2 and proba < 0.3:
        total_counts[2] += 1
        if has_muta == 1:
            counts[2] += 1
    elif proba >= 0.1 and proba < 0.2:
        total_counts[1] += 1
        if has_muta == 1:
            counts[1] += 1
    elif proba < 0.1:
        total_counts[0] += 1
        if has_muta == 1:
            counts[0] += 1
fin.close()

for x in range(BIN_SIZE):
    if total_counts[x] > 0:
        has_muta_percent = counts[x] / total_counts[x] * 100
        print('%.2f%% or %d\\%d sites in bin %d contain a mutation.' %
            (has_muta_percent, counts[x], total_counts[x], x))
    else:
        print('There are no sites in bin %d.' % x)