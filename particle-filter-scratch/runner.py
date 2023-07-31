
import numpy as np
import scipy.stats
import pypfilt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as matplotlib
matplotlib.use('QtAgg')
import pandas as pd
import plotnine as p9
from plotnine import ggplot, geom_ribbon, aes, scale_fill_gradient, \
    geom_point, geom_line, geom_vline, geom_hline, theme_bw, theme, \
    element_blank, scale_fill_discrete, scale_x_continuous, labs, \
    scale_y_continuous, scale_shape_manual


scenario_file = 'birth-death.toml'
instances = list(pypfilt.load_instances(scenario_file))
sim_instance = instances[0]
my_time_scale = sim_instance.time_scale()
num_reps = 10
my_obs_tables = pypfilt.simulate_from_model(sim_instance, particles = num_reps)

for (obs_unit, obs_table) in my_obs_tables.items():
    out_file = f'ode-bd-example-{obs_unit}.ssv'
    pypfilt.io.write_table(out_file, obs_table, my_time_scale)
