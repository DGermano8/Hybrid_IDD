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
import pdb
import src.tiv
from importlib import reload

reload(src.tiv)

def run_simulation(instance):
    """
    """
    raise Exception("Not implemented")
    # num_reps = instance.settings['num_replicates']
    # my_obs_tables = pypfilt.simulate_from_model(
    #     instance, particles = num_reps
    # )
    # sim_df = pd.DataFrame(my_obs_tables['x'])
    # sim_df['particle'] = np.tile(
    #     np.arange(num_reps),
    #     sim_df.shape[0] // num_reps
    # )
    # return sim_df


def simulation_plot(sim_df):
    """
    """
    raise Exception("Not implemented")
    # return (ggplot()
    #         + geom_line(data = sim_df,
    #                     mapping = aes(x = "time",
    #                                   y = "value",
    #                                   group = "particle"),
    #                     alpha = 0.3)
    #         + scale_y_sqrt()
    #         + labs(title = name,
    #                y = "Population size",
    #                x = "Time")
    #         + theme_bw())


def main():
    """
    """
    scenario_file = 'tiv-simulation.toml'
    insts = list(pypfilt.load_instances(scenario_file))
    sim_df = run_simulation(insts[0])
    cool_plot = simulation_plot(sim_df)
    return None


if __name__ == '__main__':
    main()

scenario_file = 'tiv-simulation.toml'
inst = list(pypfilt.load_instances(scenario_file))[0]
# sim_df = run_simulation(inst)
# cool_plot = simulation_plot(sim_df)

instance = inst
num_reps = instance.settings['num_replicates']
my_obs_tables = pypfilt.simulate_from_model(
    instance, particles = num_reps
)

sim_df = pd.DataFrame({
    'time': my_obs_tables['V']['time'],
    'V': my_obs_tables['V']['value'],
    'T': my_obs_tables['T']['value']
})


tcid_df = pd.read_csv("data/tcid.csv")
tcid_df["log10_tcid"] = (tcid_df["log10_tcid"]
                         .astype(str)
                         .apply(lambda x: float(x[2:]) if x.startswith("<=") else float(x)))
tcid_df["is_truncated"] = tcid_df["log10_tcid"] <= 0.5


plot_df = pd.melt(sim_df, id_vars = ['time'], value_vars = ['V', 'T'])

foo = (ggplot()
 + geom_line(
     data = plot_df,
     mapping = aes(x = "time",
                     y = "value",
                        group = "variable",
                        linetype = "variable"))
 + geom_point(
        data = tcid_df[tcid_df["patient"] == 1],
        mapping = aes(x = "day",
                        y = "log10_tcid",
                        shape = "is_truncated"))
    + labs(title = "Viral load (patient 1)",
           y = "",
                x = "Days post infection")
    + theme_bw()
 + theme(legend_position = "none"))

# Paper sizes must be specified in inches for this.
foo.save("out/baccam-fit.png",
        # height = 5.8, width = 8.3, # A5
        height = 4.1, width = 5.8 # A6
        # height = 2.9, width = 4.1, # A7
         )
