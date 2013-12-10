#!/usr/bin/env python
import sys

from simulation_model import SimulationModel


# run python simulation_driver.py <output.txt> <#samples> <soma muta rate>
filename = sys.argv[1]
exp_count = int(sys.argv[2])
soma_muta_rate = float(sys.argv[3]) if len(sys.argv) == 4 else 0.00000002  # default
SimulationModel.write_proba(filename, exp_count, soma_muta_rate)