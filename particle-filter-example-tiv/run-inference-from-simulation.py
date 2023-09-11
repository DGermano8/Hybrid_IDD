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
from typing import List, Dict, Any, Optional
import numpy as np
import scipy.stats              # type: ignore
import pypfilt                  # type: ignore
import matplotlib.pyplot as plt # type: ignore
import matplotlib.image as mpimg # type: ignore
import matplotlib as matplotlib # type: ignore
matplotlib.use('QtAgg')
import pandas as pd
import plotnine as p9
from plotnine import ggplot, geom_rect, aes, geom_ribbon, geom_point, scale_y_log10, scale_x_continuous, labs, theme_bw, geom_vline
import pdb

#
#
#  ********************
#  *                  *
#  * Helper functions *
#  *                  *
#  ********************
#
#
def plottable_model_cis(model_ci_df : pd.DataFrame) -> pd.DataFrame:
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

    left_blocks : List[Dict] = []
    right_blocks : List[Dict] = []
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


def const_params_from_prior(prior : Dict) -> Dict[str, float]:
    """
    Extract the constant parameters used in the simulation from the
    prior.
    """
    param_names = ["V0", "beta", "p", "c", "gamma"]
    is_const = lambda n: prior[n]["name"] == "constant"
    return {p : prior[p]["args"]["value"]
            for p in param_names if is_const(p) }


def marginals_from_prior(prior : Dict) -> Dict[str, Any]:
    """
    Extract the marginal distributions used in the inference from the
    prior.
    """
    param_names = ["V0", "beta", "p", "c", "gamma"]
    is_marginal = lambda n: prior[n]["name"] != "constant"
    return {p : prior[p]
            for p in param_names if is_marginal(p) }


def param_plt_p9(plt_df: pd.DataFrame,
                 true_value: float,
                 prior: Dict,
                 param_name: Optional[str]) -> p9.ggplot:
    """
    Plot the posterior distribution of the parameter as described by
    the given data frame.

    Note that this currently only works for *uniform* priors.
    """
    param_p9 = (ggplot()
                + geom_rect(
                    data = plt_df,
                    mapping = aes(xmin = 'ymin',
                                  xmax = 'ymax',
                                  ymin = 'xmin',
                                  ymax = 'xmax')
                )
                + geom_vline(xintercept = true_value,
                                color = 'red')
                )

    if prior["name"] == "uniform":
        min_val = prior["args"]["loc"]
        max_val = prior["args"]["scale"]
        support = (min_val, max_val)
        param_p9 = (param_p9
                    + geom_vline(xintercept = min_val,
                                 color = 'red',
                                 linetype = 'dashed')
                    + geom_vline(xintercept = max_val,
                                 color = 'red',
                                 linetype = 'dashed'))

    if param_name is not None:
        param_p9 = (param_p9 +
                    labs(title = "Posterior distribution of " + param_name))

    return ( param_p9 + theme_bw() )


def state_plt_p9(post_df: pd.DataFrame,
                 obs_df: pd.DataFrame) -> p9.ggplot:
    """
    Plot the posterior distribution of the state as described by the
    data frame of posterior intervals and the actual observations.
    """
    return (ggplot()
            + geom_ribbon(
                data = post_df,
                mapping = aes(x = 'time', ymin = 'ymin',
                              ymax = 'ymax', group = "prob"),
                alpha = 0.1
            )
            + geom_point(
                data = obs_df,
                mapping = aes(x = 'time', y = 'y'),
                color = 'red'
            )
            + scale_y_log10(name = "Viral load")
            + scale_x_continuous(name = "Time post infection (days)")
            + labs(title = "State trajectory",
                   caption = "Inference on simulated data")
            + theme_bw())


def main():
    out_dir = "out"
    in_toml = "tiv-inference-from-simulation.toml"
    #
    # Define parameter and variable names
    #
    param_names = ['V0', 'beta', 'p', 'c', 'gamma']
    state_names = ['T', 'I', 'V']
    inst_dict = {
        x.scenario_id: x
        for x in pypfilt.load_instances(in_toml)
       }
    obs_ssv = inst_dict['simulation'].settings['sim_output_file']
    #
    # Simulate some observations
    #
    sim_params = const_params_from_prior(inst_dict['simulation'].settings["prior"])
    sim_result = pypfilt.simulate_from_model(inst_dict['simulation'])
    obs_df = pd.DataFrame(sim_result['V'])
    obs_df.to_csv(obs_ssv, sep = ' ', index = False)
    #
    # Run the particle filter over simulated data
    #
    inf_ctx = inst_dict['inference'].build_context()
    mrgs = marginals_from_prior(inf_ctx.settings['prior'])
    end_time = inf_ctx.settings['time']['until']
    fit_result = pypfilt.fit(inf_ctx, filename=None)
    pst_df = pd.DataFrame(fit_result.estimation.tables['model_cints'])
    #
    # Plot the prior-posterior distributions for parameters
    #
    pst_param_df = pst_df[pst_df['time'] == end_time]
    pst_param_df = pst_param_df[pst_param_df['name'].isin(param_names)]
    pst_param_df = pst_param_df[['prob','ymin', 'ymax', 'name']]
    for param in param_names:
        dd = pst_param_df[pst_param_df['name'] == param]
        plt_df = plottable_model_cis(dd)
        param_p9 = param_plt_p9(plt_df, sim_params[param], mrgs[param],
                                param)
        param_p9.save(f"{out_dir}/demo-param-{param}-histogram.png",
                      height = 5.8, width = 8.3)
    #
    # Plot the state trajectory
    #
    pst_state_df = pst_df[pst_df['name'].isin(state_names)]
    pst_state_df = pst_state_df[['time', 'prob','ymin', 'ymax', 'name']]
    plt_df = pst_state_df[pst_state_df['name'] == 'V']
    plt_df_obs = obs_df.copy()
    plt_df_obs['y'] = 10**plt_df_obs['value']
    state_p9 = state_plt_p9(plt_df, plt_df_obs)
    state_p9.save(f"{out_dir}/demo-state-trajectory.png",
            height = 4.1, width = 5.8)


if __name__ == "__main__":
    main()
