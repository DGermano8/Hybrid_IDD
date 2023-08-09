

import scipy.stats
import numpy as np
from pypfilt.model import Model
from pypfilt.obs import Univariate, Obs
import pdb

# --------------------------------------------------------------------
# Define the process models
#
# - BirthDeathODE :: ODE
# - BirthDeathSDE :: SDE
# - BirthDeathCTMC :: CTMC
#
# --------------------------------------------------------------------

class BirthDeathODENotVec(Model):
    """
    A simple birth-death process model which has been implemented with
    a loop over the particles rather than being vectorised.
    """
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
        for p_ix in range(ctx.settings['num_replicates']):
            vec['x'][p_ix] = prior['x'][p_ix]
            vec['birthRate'][p_ix] = prior['birth'][p_ix]
            vec['deathRate'][p_ix] = prior['death'][p_ix]

    def update(self, ctx, time_step, is_forecast, prev, curr):
        """
        Update the state vectors.
        """
        deriv = np.zeros(curr['birthRate'].shape)
        for p_ix in range(ctx.settings['num_replicates']):
            curr['birthRate'][p_ix] = prev['birthRate'][p_ix]
            curr['deathRate'][p_ix] = prev['deathRate'][p_ix]
            deriv = (prev['birthRate'][p_ix] - prev['deathRate'][p_ix]) * prev['x'][p_ix]
            curr['x'][p_ix] = prev['x'][p_ix] + time_step.dt * deriv


class BirthDeathODE(Model):
    """
    A simple birth-death process model which has been implemented with
    vectorisation over the particles.
    """
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


class BirthDeathSDE(Model):
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


class BirthDeathCTMC(Model):
    def field_types(self, ctx):
        return [('birthRate', np.dtype(float)),
                ('deathRate', np.dtype(float)),
                ('x', np.dtype(int)),
                ('next_event', np.int_),
                ('next_time', np.float_)]

    def init(self, ctx, vec):
        prior = ctx.data['prior']
        vec['x'] = prior['x']
        vec['birthRate'] = prior['birth']
        vec['deathRate'] = prior['death']
        vec['next_time'] = 0
        vec['next_event'] = 0
        self.select_next_event(ctx, vec, stop_time=0)

    def update(self, ctx, time_step, is_forecast, prev, curr):
        curr[:] = prev[:]
        active = self.active_particles(curr, time_step.end)
        while any(active):
            births = np.logical_and(active, curr['next_event'] == 0)
            curr['x'][births] += 1
            deaths = np.logical_and(active, curr['next_event'] == 1)
            curr['x'][deaths] -= 1
            self.select_next_event(ctx, curr, stop_time=time_step.end)
            active = self.active_particles(curr, time_step.end)

    def active_particles(self, vec, stop_time):
        return np.logical_and(
            vec['next_time'] <= stop_time,
            vec['x'] > 0,
        )

    def select_next_event(self, ctx, vec, stop_time):
        active = self.active_particles(vec, stop_time)
        if not any(active):
            return

        x = vec['x'][active]
        birth = vec['birthRate'][active]
        death = vec['deathRate'][active]

        birth_rate = birth * x
        death_rate = death * x
        rate_sum = birth_rate + death_rate

        rng = ctx.component['random']['model']
        dt = - np.log(rng.random(x.shape)) / rate_sum
        vec['next_time'][active] += dt

        threshold = rng.random(x.shape) * rate_sum
        death_event = threshold > birth_rate
        vec['next_event'][active] = death_event.astype(np.int_)


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
