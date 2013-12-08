#!/usr/bin/env python
"""
Script to parse simulation input file. The first column represents the
probability of mutation. The second column represents if the site contains a
mutation either True or False. Each site is placed on a new line.

This script bins all sites with a probability greater than 0.9 and calculates
the percentage of these sites that contain a real mutation, meaning the value in
the has mutation column is True.
"""
import sys


fin = open(sys.argv[1])
lines = fin.readlines()
total_count = 0
count = 0
for line in lines:
    values = line.strip('\n').split('\t')
    proba = float(values[0])
    has_muta = values[1]
    if proba > 0.9:
        total_count += 1
        if has_muta == 'True':
            count += 1
fin.close()

has_muta_percent = count / total_count * 100
print('%.2f%% of %d bin sites contain a mutation.' % (has_muta_percent, total_count))