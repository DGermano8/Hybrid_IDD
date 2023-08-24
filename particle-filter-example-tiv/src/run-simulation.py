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
import tiv
from importlib import reload

reload(tiv)

def run_simulation(instance):
    """
    Runs a simulation from a given instance.

    Parameters
    ----------
    instance : pypfilt.Instance
        The instance to run the simulation from.

    Returns
    -------
    sim_df : pandas.DataFrame
        A dataframe containing the simulation results.
    """
    num_reps = instance.settings['num_replicates']
    my_obs_tables = pypfilt.simulate_from_model(
        instance, particles = num_reps
    )

    sim_df = pd.DataFrame({
        'time': my_obs_tables['V']['time'],
        'V': my_obs_tables['V']['value'],
        'T': my_obs_tables['T']['value']
    })
    return sim_df


def read_data(filename):
    tcid_df = pd.read_csv("data/tcid.csv")
    tcid_df["log10_tcid"] = (tcid_df["log10_tcid"]
                             .astype(str)
                             .apply(lambda x:
                                    float(x[2:])
                                    if x.startswith("<=")
                                    else float(x)))
    tcid_df["is_truncated"] = tcid_df["log10_tcid"] <= 0.5
    return tcid_df


def simulation_plot(sim_df, tcid_df):
    """
    Plots the simulation results.

    Parameters
    ----------
    sim_df : pandas.DataFrame
        A dataframe containing the simulation results.

    Returns
    -------
    plot : plotnine.ggplot
        The plot.
    """
    plot_df = pd.melt(sim_df,
                      id_vars = ['time'],
                      value_vars = ['V', 'T'])

    p9 = (ggplot()
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
    return p9


def main():
    """
    """
    scenario_file = 'tiv-simulation.toml'
    inst = list(pypfilt.load_instances(scenario_file))[0]
    sim_df = run_simulation(inst)
    data_df = read_data("data/tcid.csv")
    cool_plot = simulation_plot(sim_df, data_df)

    cool_plot.save("out/baccam-fit.png",
                   # height = 5.8, width = 8.3, # A5
                   height = 4.1, width = 5.8 # A6
                   # height = 2.9, width = 4.1, # A7
                   )
    return None


if __name__ == '__main__':
    main()
