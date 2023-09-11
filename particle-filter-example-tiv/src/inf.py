from typing import List, Dict, Any, Optional
import numpy as np
import scipy.stats              # type: ignore
import pypfilt                  # type: ignore
import matplotlib.pyplot as plt # type: ignore
import matplotlib.image as mpimg # type: ignore
import matplotlib as matplotlib # type: ignore
import pandas as pd
import plotnine as p9
from plotnine import ggplot, geom_rect, aes, geom_ribbon, geom_point, scale_y_log10, scale_x_continuous, labs, theme_bw, geom_vline
import pdb


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
