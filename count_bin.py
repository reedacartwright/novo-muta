#!/usr/bin/env python
"""
Script to parse input file where it is a bin subset containing every 10th line
in the original file. The first column represents the probability of mutation.
The second column represents if the site contains a mutation either True or
False. Each site is placed on a new line.

This script calculates the percentage of the sites that contain a mutation,
meaning the value in the has mutation column is True.
"""
import sys


fin = open(sys.argv[1])
lines = fin.readlines()
seq_count = len(lines)
count = 0
for line in lines:
	values = line.strip('\n').split('\t')
	if values[1] == 'True':
		count += 1
fin.close()

has_muta_percent = count / seq_count * 100
print('%.2f\%% of sites contain a mutation.' % has_muta_percent)