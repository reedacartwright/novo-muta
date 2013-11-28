#!/usr/bin/env python
"""
Driver file for TrioModel. This accepts 3 input pileup files as described here:
http://samtools.sourceforge.net/pileup.shtml

The pileup data is read and parsed into read counts that the TrioModel can
process. Default values are used for all parameters.

Assume that all pileup files have the same number of sites.
"""
import sys

import pileup


# run python pileup_driver.py <child.pileup> <mother.pileup> <father.pileup> <output.txt>
child = open(sys.argv[1])
mother = open(sys.argv[2])
father = open(sys.argv[3])
filename = sys.argv[4] if sys.argv[4] is not None else pileup.DEFAULT_FILENAME

child_lines = child.readlines()
mother_lines = mother.readlines()
father_lines = father.readlines()

child.close()
mother.close()
father.close()

pileup.writeProba(child_lines, mother_lines, father_lines, filename)