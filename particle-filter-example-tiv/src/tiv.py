import scipy.stats              # type: ignore
import numpy as np
import pypfilt                  # type: ignore
from pypfilt.model import Model # type: ignore
from pypfilt.obs import Univariate, Obs # type: ignore
import pdb
import src.JSF_Solver_BasePython as JSF

class TIV_JSF(Model):
    """
    """

    num_particles = -1
    threshold = 50
    _nu_reactants = [
        [1, 0, 1],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]
    _nu_products = [
        [0, 1, 1],
        [0, 0, 0],
        [0, 1, 1],
        [0, 0, 0],
    ]

    _nu = [[a - b for a, b in zip(r1, r2)]
           for r1, r2 in zip(_nu_products, _nu_reactants)]
    _stoich = {'nu': _nu,
               'DoDisc': [0, 1, 1],
               'nuReactant': _nu_reactants,
               'nuProduct': _nu_products}

    def field_types(self, ctx):
        """
        """
        return [
            ('lnV0', np.float_),
            ('beta', np.float_),
            ('p', np.float_),
            ('c', np.float_),
            ('gamma', np.float_),
            ('T', np.float_),
            ('I', np.float_),
            ('V', np.float_),
            ('next_event', np.int_),
            ('next_time', np.float_),
        ]

    def init(self, ctx, vec):
        """
        """
        prior = ctx.data['prior']
        self.num_particles = prior['T0'].shape[0]

        vec['beta'] = prior['beta']
        vec['p'] = prior['p']
        vec['lnV0'] = prior['lnV0']
        vec['c'] = prior['c']
        vec['gamma'] = prior['gamma']
        vec['T'] = prior['T0']
        vec['I'] = prior['I0']
        vec['V'] = np.exp(vec['lnV0'])

    def _rates(self, x, theta, time):
        """
        """
        t = x[0]
        i = x[1]
        v = x[2]
        m_beta = theta[0]
        m_p = theta[1]
        m_c = theta[2]
        m_gamma = theta[3]
        return [m_beta*(t*v),
                m_gamma*i,
                m_p*i,
                m_c*v]

    def update(self, ctx, time_step, is_forecast, prev, curr):
        """
        """
        _my_opts = {'EnforceDo': [0, 0, 0],
                    'dt': time_step.dt,
                    'SwitchingThreshold': [self.threshold,
                                           self.threshold,
                                           self.threshold]}

        for p_ix in range(self.num_particles):

            ptcl = prev[p_ix].copy()
            x0 = [ptcl['T'], ptcl['I'], ptcl['V']]
            theta = [
                ptcl['beta'],
                ptcl['p'],
                ptcl['c'],
                ptcl['gamma'],
            ]

            pdb.set_trace()
            xs, ts = JSF.JumpSwitchFlowSimulator(
                x0,
                lambda x, time: self._rates(x, theta, time),
                self._stoich,
                time_step.dt,
                _my_opts
            )

            ptcl['T'] = xs[0][-1]
            ptcl['I'] = xs[1][-1]
            ptcl['V'] = xs[2][-1]
            curr[p_ix] = ptcl

class TIV_ODE(Model):
    """
    """
    def field_types(self, ctx):
        """
        """
        return [
            ('lnV0', np.float_),
            ('beta', np.float_),
            ('p', np.float_),
            ('c', np.float_),
            ('gamma', np.float_),
            ('T', np.float_),
            ('I', np.float_),
            ('V', np.float_),
        ]

    def init(self, ctx, vec):
        """
        """
        prior = ctx.data['prior']

        vec['beta'] = prior['beta']
        vec['p'] = prior['p']
        vec['lnV0'] = prior['lnV0']
        vec['c'] = prior['c']
        vec['gamma'] = prior['gamma']
        vec['T'] = prior['T0']
        vec['I'] = prior['I0']
        vec['V'] = np.exp(vec['lnV0'])


    def update(self, ctx, time_step, is_forecast, prev, curr):
        """
        """
        curr['beta'] = prev['beta']
        curr['p'] = prev['p']
        curr['lnV0'] = prev['lnV0']
        curr['c'] = prev['c']
        curr['gamma'] = prev['gamma']

        dT_dt = - prev['beta'] * prev['V'] * prev['T']
        dI_dt = prev['beta'] * prev['V'] * prev['T'] - prev['gamma'] * prev['I']
        dV_dt = prev['p'] * prev['I'] - prev['c'] * prev['V']

        curr['T'] = prev['T'] + time_step.dt * dT_dt
        curr['I'] = prev['I'] + time_step.dt * dI_dt
        curr['V'] = prev['V'] + time_step.dt * dV_dt

    def can_smooth(self):
        """
        """
        return {'lnV0', 'beta', 'p', 'c', 'gamma'}


class BackcastStateCIs(pypfilt.summary.BackcastPredictiveCIs):
    """
    Summary statistic for the smoothing problem.
    """
    def n_rows(self, ctx, forecasting):
        n_obs_models = len(ctx.component['obs'])
        n_backcast_times = ctx.summary_count()
        return len(self._BackcastPredictiveCIs__probs) * n_backcast_times * n_obs_models


class PerfectMeasurement(Univariate):
    """
    Measurement model for perfect measurements.
    """

    def distribution(self, ctx, snapshot):
        expect = np.log10(snapshot.state_vec[self.unit])
        return scipy.stats.norm(loc=expect, scale=0.0)


class Gaussian(Univariate):
    """
    Measurement model for Gaussian measurements.
    """

    def distribution(self, ctx, snapshot):
        loc = np.log10(snapshot.state_vec[self.unit])
        scale = ctx.settings['observations'][self.unit]['scale']
        return scipy.stats.norm(loc=loc, scale=scale)
