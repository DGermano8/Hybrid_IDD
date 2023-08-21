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
# - BirthDeathHybridClock :: JSF (not vectorised)
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
        return [('birthRate', np.float_),
                ('deathRate', np.float_),
                ('x', np.int_),
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
        return [('birthRate', np.float_),
                ('deathRate', np.float_),
                ('x', np.int_),
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

    threshold = 50

    def field_types(self, ctx):
        return [('birthRate', np.float_),
                ('deathRate', np.float_),
                ('x', np.float_),
                ('next_event', np.int_),
                ('next_time', np.float_)]

    def init(self, ctx, vec):
        prior = ctx.data['prior']
        num_particles = prior['x'].shape[0]
        for p_ix in range(num_particles):
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
        num_particles = prev['x'].shape[0]
        for p_ix in range(num_particles):
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
                (curr_ptcl['next_time'] < time_step.end)):
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
        assert remaining_time > 0

        deriv = (ptcl['birthRate'] - ptcl['deathRate']) * ptcl['x']
        ptcl['x'] += remaining_time * deriv
        ptcl['next_time'] += remaining_time


class BirthDeathHybridClock(Model):
    """
    This is similar to the BirthDeathHybrid class, but makes use of
    the formulation as differential equations with events (a.k.a.,
    event clocks) so is closer to a general solution.

    As with the previous hybrid example, this is *not* vectorised,
    instead it loops over the particles to keep the code simpler.
    """

    threshold = 50

    def field_types(self, ctx):
        # The _jump_clock_u variable is used to store the initial
        # value of the jump clock because this is needed to update the
        # clock properly.
        return [('birthRate', np.float_),
                ('deathRate', np.float_),
                ('x', np.float_),
                ('jump_clock', np.float_),
                ('_jump_clock_u', np.float_),
                ('jump_clock_type', np.int_),
                ('ptcl_time', np.float_)]

    def init(self, ctx, vec):
        prior = ctx.data['prior']
        num_particles = prior['x'].shape[0]
        for p_ix in range(num_particles):
            vec['x'][p_ix] = prior['x'][p_ix]
            vec['birthRate'][p_ix] = prior['birth'][p_ix]
            vec['deathRate'][p_ix] = prior['death'][p_ix]
            vec['jump_clock'][p_ix] = np.infty
            vec['_jump_clock_u'][p_ix] = -0.0
            vec['jump_clock_type'][p_ix] = -1
            vec['ptcl_time'][p_ix] = 0
            self.reset_clocks(ctx, vec[p_ix])

    def update(self, ctx, time_step, is_forecast, prev, curr):
        num_particles = prev['x'].shape[0]
        for p_ix in range(num_particles):
            # print('Particle %d' % p_ix)
            curr_ptcl = prev[p_ix].copy()
            while ((curr_ptcl['ptcl_time'] < time_step.end) and
                   (curr_ptcl['x'] > 0)):
                # print('Time %f' % curr_ptcl['ptcl_time'])
                self.euler_step(ctx, curr_ptcl, time_step.end)
                if (curr_ptcl['jump_clock'] <= 0):
                    jump_time = self.jump_clock_root(prev[p_ix], curr_ptcl)
                    curr_ptcl = prev[p_ix].copy()

                    self.euler_step(ctx, curr_ptcl, jump_time)
                    # pdb.set_trace()
                    if curr_ptcl['jump_clock_type'] == 0:
                        curr_ptcl['x'] -= 1
                    elif curr_ptcl['jump_clock_type'] == 1:
                        curr_ptcl['x'] += 1
                    else:
                        raise ValueError('Invalid event')
                    self.reset_clocks(ctx, curr_ptcl)

            curr[p_ix] = curr_ptcl

    def jump_clock_root(self, prev_ptcl, curr_ptcl):
        time_a = prev_ptcl['ptcl_time']
        jc_a = prev_ptcl['jump_clock']
        time_b = curr_ptcl['ptcl_time']
        jc_b = curr_ptcl['jump_clock']

        assert jc_a > 0
        assert jc_b < 0
        assert time_a < time_b

        return time_a + (time_b - time_a) * jc_a / (jc_a - jc_b)

    def reset_clocks(self, ctx, ptcl):
        if ptcl['x'] <= 0:
            return

        rng = ctx.component['random']['model']
        ptcl['jump_clock'] = rng.random((1,))
        ptcl['_jump_clock_u'] = ptcl['jump_clock']
        ind_rate = ptcl['birthRate'] + ptcl['deathRate']
        is_birth = (rng.random((1,)) * ind_rate) < ptcl['birthRate']
        ptcl['jump_clock_type'] = is_birth.astype(np.int_)

    def euler_step(self, ctx, ptcl, stop_time):
        """
        Perform an Euler step up until the stop time.
        """
        remaining_time = stop_time - ptcl['ptcl_time']
        assert remaining_time >= 0
        ptcl['ptcl_time'] += remaining_time

        if ptcl['x'] <= 0:
            ptcl['jump_clock'] = np.infty
            return

        if ptcl['x'] <= self.threshold:
            j_u = ptcl['_jump_clock_u']
            j_t = ptcl['jump_clock']
            net_rate = (ptcl['birthRate'] + ptcl['deathRate']) * ptcl['x']
            exp_of_intgrl = np.exp(- net_rate * remaining_time)
            ptcl['jump_clock'] = j_u - (1 - (j_t - j_u + 1)*exp_of_intgrl)
        else:
            deriv = (ptcl['birthRate'] - ptcl['deathRate']) * ptcl['x']
            ptcl['x'] += remaining_time * deriv
            ptcl['jump_clock'] = np.infty


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
