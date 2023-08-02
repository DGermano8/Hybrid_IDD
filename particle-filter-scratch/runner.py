
import numpy as np
import scipy.stats
import pypfilt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as matplotlib
matplotlib.use('QtAgg')
import pandas as pd
import plotnine as p9
from plotnine import *


scenario_file = 'birth-death.toml'
instance_dict = {x.scenario_id: x for x
                 in pypfilt.load_instances(scenario_file)}

def run_simulation(instance):
    num_reps = instance.settings['num_replicates']
    my_obs_tables = pypfilt.simulate_from_model(
        instance, particles = num_reps
    )
    sim_df = pd.DataFrame(my_obs_tables['x'])
    sim_df['particle'] = np.tile(
        np.arange(num_reps),
        sim_df.shape[0] // num_reps
    )
    return sim_df

def simulation_plot(sim_df, name):
    return (ggplot()
            + geom_line(data = sim_df,
                        mapping = aes(x = "time",
                                      y = "value",
                                      group = "particle"))
            + scale_y_sqrt()
            + labs(title = name))

plots = [
    simulation_plot(run_simulation(value), key)
    for key, value in instance_dict.items()]
p9.save_as_pdf_pages(plots, filename = "demo-simulations.pdf")
