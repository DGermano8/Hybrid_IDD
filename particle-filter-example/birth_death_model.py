import scipy.stats
import numpy as np
from pypfilt.model import Model
from pypfilt.obs import Univariate, Obs
import pdb

# --------------------------------------------------------------------
# Define the process models
#
# - BirthDeathODENotVec :: ODE (not vectorised)
# - BirthDeathODE :: ODE (vectorised)
# - BirthDeathSDENotVec :: SDE (not vectorised)
# - BirthDeathSDE :: SDE (vectorised)
# - BirthDeathCTMCNotVec :: CTMC (not vectorised)
# - BirthDeathCTMC :: CTMC (vectorised)
# - BirthDeathHybrid :: JSF (not vectorised)
#
# The models that have a "NotVec" suffix are not vectorised, they loop
# over the particles/replicates to update their state vectors.
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
        (Destructively) update the state vector `curr`.
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


class BirthDeathSDENotVec(Model):
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
        (Destructively) update the state vector `curr`.
        """
        rng = ctx.component['random']['model']
        for p_ix in range(ctx.settings['num_replicates']):
            curr['birthRate'][p_ix] = prev['birthRate'][p_ix]
            curr['deathRate'][p_ix] = prev['deathRate'][p_ix]
            diff = (prev['birthRate'][p_ix] - prev['deathRate'][p_ix]) * prev['x'][p_ix] * time_step.dt
            wein = np.sqrt(diff) * rng.normal(size=(1,))
            curr['x'][p_ix] = np.clip(prev['x'][p_ix] + diff + wein, 0, None)


class BirthDeathCTMCNotVec(Model):

    def field_types(self, ctx):
        return [('birthRate', np.dtype(float)),
                ('deathRate', np.dtype(float)),
                ('x', np.dtype(int)),
                ('next_event', np.int_),
                ('next_time', np.float_)]

    def init(self, ctx, vec):
        # NOTE that the `next_time` used here tracks the *cumulative*
        # time for that particular particle rather than the wait time
        # until the next event.
        prior = ctx.data['prior']
        for p_ix in range(ctx.settings['num_replicates']):
            vec['x'][p_ix] = prior['x'][p_ix]
            vec['birthRate'][p_ix] = prior['birth'][p_ix]
            vec['deathRate'][p_ix] = prior['death'][p_ix]
            vec['next_time'][p_ix] = 0
            vec['next_event'][p_ix] = 0
            self.select_next_event(ctx, vec[p_ix], stop_time=0)

    def update(self, ctx, time_step, is_forecast, prev, curr):
        for p_ix in range(ctx.settings['num_replicates']):
            curr_ptcl = prev[p_ix]
            # Since the wait times between events is stochastic we
            # need to keep looping until the next event time goes
            # beyond the end of the current time step.
            while ((curr_ptcl['next_time'] <= time_step.end) and
                   (curr_ptcl['x'] > 0)):
                if curr_ptcl['next_event'] == 0:
                    curr_ptcl['x'] -= 1
                elif curr_ptcl['next_event'] == 1:
                    curr_ptcl['x'] += 1
                else:
                    raise ValueError('Invalid event')
                self.select_next_event(ctx, curr_ptcl,
                                       stop_time=time_step.end)
            curr[p_ix] = curr_ptcl

    def select_next_event(self, ctx, ptcl, stop_time):
        # NOTE since the particle is a mutable object it should be
        # passed by reference rather than value which means we just
        # need to update it rather than return a modified copy.
        if ptcl['x'] <= 0:
            return

        rng = ctx.component['random']['model']

        ind_rate = ptcl['birthRate'] + ptcl['deathRate']
        dt = - np.log(rng.random((1,))) / (ind_rate * ptcl['x'])
        ptcl['next_time'] += dt

        is_birth = (rng.random((1,)) * ind_rate) < ptcl['birthRate']
        ptcl['next_event'] = is_birth.astype(np.int_)


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


class BirthDeathHybrid(Model):

    threshold = 1000

    def field_types(self, ctx):
        return [('birthRate', np.dtype(float)),
                ('deathRate', np.dtype(float)),
                ('x', np.dtype(int)),
                ('next_event', np.int_),
                ('next_time', np.float_)]

    def init(self, ctx, vec):
        prior = ctx.data['prior']
        for p_ix in range(ctx.settings['num_replicates']):
            vec['x'][p_ix] = prior['x'][p_ix]
            vec['birthRate'][p_ix] = prior['birth'][p_ix]
            vec['deathRate'][p_ix] = prior['death'][p_ix]
            vec['next_time'][p_ix] = 0
            vec['next_event'][p_ix] = 0
            self.select_next_event(ctx, vec[p_ix], stop_time=0)

    def update(self, ctx, time_step, is_forecast, prev, curr):
        # NOTE we are taking a short-cut here assuming that the
        # process can only go from the discrete to the continuous
        # regime and not vice versa, which is not true in general and
        # implicitly assumes that the birth-rate is not less than the
        # death-rate.
        for p_ix in range(ctx.settings['num_replicates']):
            curr_ptcl = prev[p_ix]
            while ((curr_ptcl['x'] <= self.threshold) and
                   (curr_ptcl['next_time'] <= time_step.end) and
                   (curr_ptcl['x'] > 0)):
                if curr_ptcl['next_event'] == 0:
                    curr_ptcl['x'] -= 1
                elif curr_ptcl['next_event'] == 1:
                    curr_ptcl['x'] += 1
                else:
                    raise ValueError('Invalid event')

                if curr_ptcl['x'] > self.threshold:
                    self.euler_step(ctx, curr_ptcl, time_step.end)
                else:
                    self.select_next_event(ctx, curr_ptcl, time_step.end)


            if ((curr_ptcl['x'] > self.threshold) and
                (curr_ptcl['next_time'] <= time_step.end)):
                self.euler_step(ctx, curr_ptcl, time_step.end)

            curr[p_ix] = curr_ptcl

    def select_next_event(self, ctx, ptcl, stop_time):
        """
        Select the next event for a particle along with the time of
        the event.
        """
        if ptcl['x'] <= 0:
            return

        rng = ctx.component['random']['model']

        ind_rate = ptcl['birthRate'] + ptcl['deathRate']
        dt = - np.log(rng.random((1,))) / (ind_rate * ptcl['x'])
        ptcl['next_time'] += dt

        is_birth = (rng.random((1,)) * ind_rate) < ptcl['birthRate']
        ptcl['next_event'] = is_birth.astype(np.int_)

    def euler_step(self, ctx, ptcl, stop_time):
        """
        Perform an Euler step up until the stop time.
        """
        if ptcl['x'] <= 0:
            return

        remaining_time = stop_time - ptcl['next_time']
        assert remaining_time >= 0

        deriv = (ptcl['birthRate'] - ptcl['deathRate']) * ptcl['x']
        ptcl['x'] += remaining_time * deriv
        ptcl['next_time'] += remaining_time


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
