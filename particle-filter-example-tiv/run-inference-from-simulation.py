# run-inference-from-simulation.py
#
# program: Inference on simulated data
#
# programmer: Alexander E. Zarebski
# date: 2023-08-29
#
# description: A simple example of parameter estimation using
# simulated data based on parameters from Baccam et al (2006).
#
#
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

#
#
#  ********************
#  *                  *
#  * Helper functions *
#  *                  *
#  ********************
#
#
def plottable_model_cis(model_ci_df):
    """
    Converts the model_cints table into a dataframe that can be
    plotted using plot9.

    Assumes that the model_cints table has the following columns:
    - prob (as a percentage)
    - ymin
    - ymax

    Returns a dataframe with the following columns:
    - prob
    - mass
    - ymin
    - ymax
    """
    assert (model_ci_df['prob'].dtype == np.dtype('int8') or
            model_ci_df['prob'].dtype == np.dtype('int32') or
            model_ci_df['prob'].dtype == np.dtype('int64'))
    dd = sorted(model_ci_df.to_dict(orient= 'records'),
                key=lambda k: k['prob'])

    left_blocks, right_blocks = [], []
    prev_prob = 0
    probs = [d['prob'] for d in dd]
    start_prob = np.array(probs).min()

    for prob in probs:
        di = [d for d in dd if d['prob'] == prob][0]
        left_block_right = (0.5 * ( di['ymin'] + di['ymax'] )
                            if prob == start_prob
                        else left_blocks[-1]['ymin'])
        di_left = {'prob': di['prob'],
                   'mass': 0.5 * (di['prob'] / 100 - prev_prob / 100),
                   'ymax': left_block_right,
                   'ymin': di['ymin']}
        left_blocks.append(di_left)
        right_block_left = (0.5 * ( di['ymin'] + di['ymax'] )
                        if prob == start_prob
                        else right_blocks[-1]['ymax'])
        di_right = {'prob': di['prob'],
                    'mass': 0.5 * (di['prob'] / 100 - prev_prob / 100),
                    'ymax': di['ymax'],
                    'ymin': right_block_left}
        right_blocks.append(di_right)
        prev_prob = di['prob']

    tmp = pd.concat([pd.DataFrame(left_blocks),
                     pd.DataFrame(right_blocks)])
    tmp['xmin'] = 0
    tmp['xmax'] = tmp['mass'] / (tmp['ymax'] - tmp['ymin'])
    return tmp
#
#
#  ******************************
#  *                            *
#  * Simulate some observations *
#  *                            *
#  ******************************
#
#
inst_dict = {
    x.scenario_id: x
    for x in pypfilt.load_instances("tiv-inference-from-simulation.toml")
}
sim_result = pypfilt.simulate_from_model(inst_dict['simulation'])
obs_df = pd.DataFrame(sim_result['V'])
obs_ssv = inst_dict['inference'].settings['observations']['V']['file']
obs_df.to_csv(obs_ssv, sep = ' ', index = False)
#
#
#  ***********************************************
#  *                                             *
#  * Run the particle filter over simulated data *
#  *                                             *
#  ***********************************************
#
#
inf_ctx = inst_dict['inference'].build_context()
end_time = inf_ctx.settings['time']['until']
fit_result = pypfilt.fit(inf_ctx, filename=None)
pst_df = pd.DataFrame(fit_result.estimation.tables['model_cints'])
#
#
#  *********************************************************
#  *                                                       *
#  * Plot the prior-posterior distributions for parameters *
#  *                                                       *
#  *********************************************************
#
#
param_names = ['V0', 'beta', 'p', 'c', 'gamma']
pst_param_df = pst_df[pst_df['time'] == end_time]
pst_param_df = pst_param_df[pst_param_df['name'].isin(param_names)]
pst_param_df = pst_param_df[['prob','ymin', 'ymax', 'name']]

for param in param_names:
    dd = pst_param_df[pst_param_df['name'] == param]
    plt_df = plottable_model_cis(dd)

    param_p9_V0 = (ggplot()
                   + geom_rect(
                       data = plt_df,
                       mapping = aes(xmin = 'ymin',
                                     xmax = 'ymax',
                                     ymin = 'xmin',
                                     ymax = 'xmax')
                      )
                      )

    param_p9_V0.save(f"demo-param-{param}-histogram.png",
                     height = 5.8, width = 8.3)
#
#
#  *****************************
#  *                           *
#  * Plot the state trajectory *
#  *                           *
#  *****************************
#
#
pst_state_df = pst_df[pst_df['name'].isin(['T', 'I', 'V'])]
pst_state_df = pst_state_df[['time', 'prob','ymin', 'ymax', 'name']]

plt_df = pst_state_df[pst_state_df['name'] == 'V']
plt_df_obs = obs_df.copy()
plt_df_obs['y'] = 10**plt_df_obs['value']

state_p9 = (ggplot()
            + geom_ribbon(
                data = plt_df,
                mapping = aes(x = 'time', ymin = 'ymin', ymax = 'ymax', group = "prob"),
                alpha = 0.1
            )
            + geom_point(
                data = plt_df_obs,
                mapping = aes(x = 'time', y = 'y'),
                color = 'red'
            )
            + scale_y_log10(name = "Viral load")
            + scale_x_continuous(name = "Time post infection (days)")
            + labs(title = "State trajectory",
                   caption = "Inference on simulated data")
            + theme_bw()
            )

state_p9.save("demo-state-trajectory.png",
        height = 4.1, width = 5.8)
