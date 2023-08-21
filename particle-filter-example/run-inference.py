
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


def birth_rate_plot(param_cris_df):
    """
    Plot the birth rate credible intervals as a function of the amount
    of data used.

    :param param_cris_df: Dataframe of birth rate credible intervals.
    :return: Plotnine plot object.
    """
    pt_est_mask = param_cris_df['prob'] == 0
    birth_rate_p9 = (ggplot()
                     + geom_ribbon(data = param_cris_df,
                                   mapping = aes(x = "time",
                                                 ymin = "ymin",
                                                 ymax = "ymax",
                                                 group = "prob"),
                                   alpha = 0.2,
                                   size = 1)
                     + geom_line(data = param_cris_df[pt_est_mask],
                                 mapping = aes(x = "time",
                                               y = "ymin"))
                     + geom_hline(yintercept = 2.0,
                                  linetype = "dashed")
                     + scale_x_continuous(name = "Amount of data used")
                     + scale_y_continuous(name = "Birth rate")
                     + labs(title = "Birth rate credible intervals")
                     + theme_bw())
    return birth_rate_p9


def posterior_dataframes(results):
    """
    Posterior dataframes from the results object.

    :param results: Results object from the inference.
    :return: Dictionary of dataframes.
    """
    param_cris_df = pd.DataFrame(results.estimation.tables['model_cints'])
    param_cris_df = param_cris_df[param_cris_df['name'] == 'birthRate']
    param_cris_df['prob'] = pd.Categorical(
        param_cris_df['prob'],
        categories=param_cris_df['prob'].unique(),
        ordered=True
       )
    return {'birth_rate_df': param_cris_df}


def main():
    """
    Main function that runs the inference demonstration.

    :return: None
    """
    scenario_file = 'birth-death-inference.toml'
    instances = list(pypfilt.load_instances(scenario_file))

    # Simulate a dataset using the simulation instance
    sim_inst = instances[0]
    obs_ssv = sim_inst.settings['observations']['x']['file']
    time_scale = sim_inst.time_scale()
    obs_tables = pypfilt.simulate_from_model(sim_inst)
    pypfilt.io.write_table(obs_ssv, obs_tables['x'], time_scale)

    # Carry out the inference using the inference instances and
    # generate plots to summarise the results.
    for inf_inst in instances[1:]:
        output_id = inf_inst.settings['output_id']
        fcst_time = inf_inst.settings['forecast_time']
        ctx = inf_inst.build_context()
        results = pypfilt.forecast(ctx, [fcst_time], filename=None)

        posterior = posterior_dataframes(results)
        birth_rate_p9 = birth_rate_plot(posterior['birth_rate_df'])
        birth_rate_p9.save(f"out/{output_id}-demo-birth-rate.png")


if __name__ == '__main__':
    main()
