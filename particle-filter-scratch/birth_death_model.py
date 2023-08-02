

import scipy.stats
import numpy as np
from pypfilt.model import Model
from pypfilt.obs import Univariate, Obs

# --------------------------------------------------------------------
# Define the process models
# --------------------------------------------------------------------

class BirthDeathEuler(Model):
    def field_types(self, ctx):
        """
        Define the state that is used to completely specify a
        particle.
        """
        return [
            ('x', np.float_),
            ('birthRate', np.float_),
            ('deathRate', np.float_),
        ]

    def init(self, ctx, vec):
        """
        Initialise the state vectors based on the prior distribution.
        """
        prior = ctx.data['prior']
        vec['x'] = prior['x']
        vec['birthRate'] = prior['birth']
        vec['deathRate'] = prior['death']

    def update(self, ctx, time_step, is_forecast, prev, curr):
        """
        Update the state vectors.
        """
        curr['birthRate'] = prev['birthRate']
        curr['deathRate'] = prev['deathRate']
        deriv = (prev['birthRate'] - prev['deathRate']) * prev['x']
        curr['x'] = prev['x'] + time_step.dt * deriv


class BirthDeathNoisy(Model):
    def field_types(self, ctx):
        """
        Define the state that is used to completely specify a
        particle.
        """
        return [
            ('x', np.float_),
            ('birthRate', np.float_),
            ('deathRate', np.float_),
        ]

    def init(self, ctx, vec):
        """
        Initialise the state vectors based on the prior distribution.
        """
        prior = ctx.data['prior']
        vec['x'] = prior['x']
        vec['birthRate'] = prior['birth']
        vec['deathRate'] = prior['death']

    def update(self, ctx, time_step, is_forecast, prev, curr):
        """
        (Destructively) update the state vector `curr`.
        """
        rng = ctx.component['random']['model']
        curr['birthRate'] = prev['birthRate']
        curr['deathRate'] = prev['deathRate']
        diff = (prev['birthRate'] - prev['deathRate']) * prev['x'] * time_step.dt
        wein = np.sqrt(diff) * rng.normal(size=prev['x'].shape)
        curr['x'] = np.clip(prev['x'] + diff + wein, 0, None)

# --------------------------------------------------------------------
# Define the observation models
# --------------------------------------------------------------------

class UniformObservation(Univariate):
    """
    Observation without error.
    """
    def distribution(self, ctx, snapshot):
        expected_value = snapshot.state_vec['x']
        return scipy.stats.randint(low=np.round(expected_value),
                                   high=np.round(expected_value+1))

class NoisyStateObservation(Univariate):
    """
    Observation with Poisson noise.
    """
    def distribution(self, ctx, snapshot):
        expected_value = snapshot.state_vec['x']
        return scipy.stats.poisson(mu=expected_value)
