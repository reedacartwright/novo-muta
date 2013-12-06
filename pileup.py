"""
Pileup class contains methods to parse and align the sequences in pileup files,
and creates a TrioModel object from the read counts and writes the probabilities
to a text file.
"""
import itertools
import sys

from family.trio_model import TrioModel
from family import utilities as ut

THRESHOLD = 0.01


def trim_header(lines):
    """
    Remove initial sequences that are not necessary.
    Assume there are sequences that are not N.

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
            return lines[count:]

def get_reads(line):
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

def write_proba(child, mother, father, filename):
    """
    Write probabilites of each site on a new line to a text file.

    Args:
        child: Contents of the child pileup file.
        mother: Contents of the mother pileup file.
        father: Contents of the father pileup file.
        filename: String name of file to be written.
    """
    child_lines = trim_header(child)
    mother_lines = trim_header(mother)
    father_lines = trim_header(father)
    fout = open(filename, 'w')
    for c, m, f in itertools.izip_longest(child_lines, mother_lines, father_lines):
        reads = [get_reads(c), get_reads(m), get_reads(f)]
        trio_model = TrioModel(reads=reads)
        proba = trio_model.trio()
        if proba >= THRESHOLD:
            site_proba = "%s\t%s\n" % (c.split("\t")[1], str(proba))
            fout.write(site_proba)
    fout.close()