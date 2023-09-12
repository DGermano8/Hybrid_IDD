import scipy.stats
import numpy as np
from pypfilt.model import Model
from pypfilt.obs import Univariate, Obs
import pdb

class TIV_ODE(Model):
    """
    """
    def field_types(self, ctx):
        """
        """
        return [
            ('V0', np.float_),
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

        vec['V0'] = prior['V0']
        vec['beta'] = prior['beta']
        vec['p'] = prior['p']
        vec['c'] = prior['c']
        vec['gamma'] = prior['gamma']
        vec['T'] = prior['T0']
        vec['I'] = prior['I0']
        vec['V'] = vec['V0']


    def update(self, ctx, time_step, is_forecast, prev, curr):
        """
        """
        curr['V0'] = prev['V0']
        curr['beta'] = prev['beta']
        curr['p'] = prev['p']
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
        return {'V0', 'beta', 'p', 'c', 'gamma'}


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
