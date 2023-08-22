import scipy.stats
import numpy as np
from pypfilt.model import Model
from pypfilt.obs import Univariate, Obs
import pdb


def _update(ctx, time_step, ptcl):
    """
    """
    while ((ptcl['next_time'] <= time_step.end) and
           (ptcl['I'] > 0)):
        if ptcl['next_event'] == 0:
            ptcl['S'] += 1
            ptcl['I'] -= 1
        elif ptcl['next_event'] == 1:
            ptcl['S'] -= 1
            ptcl['I'] += 1
        else:
            raise ValueError('Invalid event')
        _select_next_event(ctx, ptcl, stop_time=time_step.end)

def _select_next_event(ctx, ptcl, stop_time):
    """
    """
    if ptcl['I'] == 0:
        return

    rng = ctx.component['random']['model']

    inf_rate = ptcl['betaCoef'] * ptcl['S'] * ptcl['I'] / ptcl['N']
    rec_rate = ptcl['gammaCoef'] * ptcl['I']
    net_event_rate = inf_rate + rec_rate
    dt = - np.log(rng.random((1,))) / net_event_rate
    ptcl['next_time'] += dt

    is_inf = (rng.random((1,)) * net_event_rate) < inf_rate
    ptcl['next_event'] = is_inf.astype(np.int_)


def magic(ctx, time_step, input_ptcl):
    ptcl = input_ptcl.copy()
    if np.isnan(ptcl['next_time']):
        ptcl['next_time'] = 0
        _select_next_event(ctx, ptcl, stop_time=0)

    _update(ctx, time_step, ptcl)
    return ptcl
