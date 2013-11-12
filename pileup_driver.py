#!/usr/bin/env python
"""
Driver file for TrioModel. This accepts 3 input pileup files as described here:
http://samtools.sourceforge.net/pileup.shtml

The pileup data is read and parsed into read counts that the TrioModel can
process. Default values are used for all parameters.

Assume that all pileup files have the same number of sites and are aligned.
"""
import sys

from family.trio_model import TrioModel
from family import utilities as ut

# run python pileup_driver.py <child.pileup> <mother.pileup> <father.pileup>
child = open(sys.argv[1])
mother = open(sys.argv[2])
father = open(sys.argv[3])

def trimHeader(lines):
    """
    Remove initial sequences that are not necessary.

    Args:
        lines: An iterable of the file that is being read.

    Returns:
        An array containing the contents of handle excluding any initial lines
        with a reference N.
    """
    count = 0
    for line in lines:
        values = line.strip("\n").split("\t")
        ref = values[2]
        if ref == 'N':
            count += 1
        else:
            break
    return lines[count:]

child_lines = trimHeader(child.readlines())
mother_lines = trimHeader(mother.readlines())
father_lines = trimHeader(father.readlines())

child.close()
mother.close()
father.close()

def getReads(line):
    """
    Parse pileup data into an array of read counts per nucleotide.

    Args:
        line: String read from a pileup file representing a single site sequence.

    Returns:
        A 4 element array containing read counts per nucleotide: [A, C, G, T].
    """
    values = line.strip("\n").split("\t")
    ref = values[2]
    bases = values[4]
    matches = bases.count(".") + bases.count(",")
    A = bases.count("A") + bases.count("a")
    C = bases.count("C") + bases.count("c")
    G = bases.count("G") + bases.count("g")
    T = bases.count("T") + bases.count("t")
    pairs = {'A': [matches, C, G, T],
             'C': [A, matches, G, T],
             'G': [A, C, matches, T],
             'T': [A, C, G, matches]}
    return pairs.get(ref)

def writeProba(filename):
    """
    Write probabilites of each site on a new line to a text file.

    Args:
        filename: String name of file to be written.
    """
    fout = open(filename, 'w')
    for c in child_lines:
        m = mother_lines.pop(0)
        f = father_lines.pop(0)
        reads = [getReads(c), getReads(m), getReads(f)]
        trio_model = TrioModel(reads=reads)
        proba = str(trio_model.trio()) + "\n"
        fout.write(proba)
    fout.close()

writeProba('pileup_proba.txt')