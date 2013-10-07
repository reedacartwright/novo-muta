#!/usr/bin/env python
"""
Driver file for TrioModel. This takes in an input text file where each parameter
is deliminated by a tab and each model object is on a new line. Leave an
optional field empty to use the default value:

#A #C # G #T child reads
#A #C # G #T mom reads
#A #C # G #T dad reads
population mutation rate
germline mutation rate
somatic mutation rate
sequencing error rate
dirichlet multinomial dispersion (optional)
dirichlet multinomial bias (optional)
"""
import sys

# TODO: when family package is finished use import family
from family.trio_model import TrioModel
from family import utilities as ut


# run python driver.py parameters.txt
handle = open(sys.argv[1])

for line in handle:
    values = line.strip("\n").split("\t")
    child_read_arr = values[:4]
    mom_read_arr = values[4:8]
    dad_read_arr = values[8:12]
    child_read = [int(count) for count in child_read_arr]
    mom_read = [int(count) for count in mom_read_arr]
    dad_read = [int(count) for count in dad_read_arr]
    reads = [child_read, mom_read, dad_read]
    rates_arr = values[12:16]
    rates = [float(rate) for rate in rates_arr]
    disp = values[13] if not values[13] else None
    bias = values[14] if not values[14] else None

    trio_model = TrioModel(
        reads=reads,
        pop_muta_rate=rates[0],
        germ_muta_rate=rates[1],
        soma_muta_rate=rates[2],
        seq_err_rate=rates[3],
        dm_disp=disp,
        dm_bias=bias
    )
    proba = trio_model.trio()
    print(proba)

handle.close()