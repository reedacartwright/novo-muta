#!/usr/bin/env python
import sys

from simulation_model import SimulationModel


# run python simulation_driver.py <output.txt> <#samples> <coverage> <germ muta rate> <soma muta rate>
filename = sys.argv[1]
exp_count = int(sys.argv[2])
cov = int(sys.argv[3])
germ_muta_rate = float(sys.argv[4]) if len(sys.argv) > 4 else None
soma_muta_rate = float(sys.argv[5]) if len(sys.argv) == 6 else None
SimulationModel.write_proba(
	filename,
	exp_count,
	cov,
	germ_muta_rate,
	soma_muta_rate
)