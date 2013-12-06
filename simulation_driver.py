#!/usr/bin/env python
import sys

from simulation_model import SimulationModel


filename = sys.argv[1]
exp_count = int(sys.argv[2])
SimulationModel.write_proba(filename, exp_count)