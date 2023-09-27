
import argparse
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
from inf import plottable_model_cis, param_plt_p9, state_plt_p9, sirs_run_inference
import os


inst = list(pypfilt.load_instances("sirs-inference.toml"))[0]

out_dir = 'outputs/' +  inst.settings['components']['model'] + '_' + str(inst.settings['filter']['particles'])

# make output directory if it doesn't exist
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

inst.settings['observations']['I']['file'] = "out/sim_df.ssv"

inf_ctx = inst.build_context()
inf_results = sirs_run_inference(inf_ctx)

# out_dir = 'ODE_Model'
# out_dir = 'JSF_Model'
# param_names = ["betaCoef", "gammaCoef", "omegaCoef", "kappaCoef", "muCoef"]
param_names = ["betaCoef", "gammaCoef", "omegaCoef",]

end_time = inf_results['end_time']

mrgs = inf_results['marginals']
pst_param_df = inf_results['posterior_param_df']
for param in param_names:
    # if cli_args['verbose']:
    #     print(f"\tGenerating plot for parameter: {param}")
    dd = pst_param_df[pst_param_df['name'] == param]
    plt_df = plottable_model_cis(dd)
    param_p9 = param_plt_p9(plt_df, None, mrgs[param],
                            param)
    param_p9.save(f"{out_dir}/demo-param-{param}-histogram.png",
                  height = 5.8, width = 8.3)

cli_args = {'obs_ssv': 'out/sim_df.ssv'}
pst_state_df = inf_results['posterior_state_df']
plt_df = pst_state_df[pst_state_df['unit'] == 'I']
plt_df['ymin'] = plt_df['ymin']
plt_df['ymax'] = plt_df['ymax']
plt_df_obs = pd.read_csv(cli_args['obs_ssv'], sep = ' ')
plt_df_obs['y'] = plt_df_obs['value']
# write the pandas dataframe to csv
plt_df.to_csv(f"{out_dir}/plt_df.csv", index=False)
state_p9 = state_plt_p9(plt_df, plt_df_obs)
state_p9.save(f"{out_dir}/demo-state-trajectory.png",
                      height = 4.1, width = 5.8)