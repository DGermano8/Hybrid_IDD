import scipy.stats
import numpy as np
from pypfilt.model import Model
from pypfilt.obs import Univariate, Obs
import pdb

import JSF_Solver_BasePython as JSF

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


_switch_thresh = [100, 100]
_nu_reactants = [[1, 1],
                 [0, 1]]
_nu_products = [[0, 2],
                [1, 0]]
_nu = [[a - b for a, b in zip(r1, r2)]
       for r1, r2 in zip(_nu_products, _nu_reactants)]
DoDisc = [1, 1]
EnforceDo = [0, 0]
_stoich = {'nu': _nu,
           'DoDisc': DoDisc,
           'nuReactant': _nu_reactants,
           'nuProduct': _nu_products}

def rates(x, theta, time):
    """
    """
    s = x[0]
    i = x[1]
    m_beta = theta[0]
    m_gamma = theta[1]
    return [m_beta*(s*i)/(s+i),
            m_gamma*i]


def magic(ctx, time_step, input_ptcl):
    _my_opts = {'EnforceDo': EnforceDo,
                'dt': time_step.dt,
                'SwitchingThreshold': _switch_thresh}

    ptcl = input_ptcl.copy()

    x0 = [ptcl['S'], ptcl['I']]
    theta = [ptcl['betaCoef'], ptcl['gammaCoef']]

    xs, ts = JSF.JumpSwitchFlowSimulator(x0,lambda x, time: rates(x, theta, time),_stoich,time_step.dt,_my_opts)

    ptcl['S'] = xs[0][-1]
    ptcl['I'] = xs[1][-1]
    # pdb.set_trace()

    return ptcl
