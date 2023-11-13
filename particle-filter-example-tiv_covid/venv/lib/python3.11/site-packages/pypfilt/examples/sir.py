"""
Multiple examples of the classic SIR model of infectious disease epidemics.
"""

import numpy as np
import pkgutil
import scipy.stats

from ..model import Model, OdeModel
from ..obs import Univariate


def sir_toml_data():
    """
    Return the contents of the example file "sir.toml".
    """
    return pkgutil.get_data('pypfilt.examples', 'sir.toml').decode()


class SirCtmc(Model):
    """
    A continuous-time Markov chain implementation of the SIR model.

    The model settings must include the following keys:

    * ``population_size``: The number of individuals in the population.
    """

    def field_types(self, ctx):
        """
        Define the state vector structure.
        """
        return [
            # Model state variables.
            ('S', np.int_), ('I', np.int_), ('R', np.int_),
            # Model parameters.
            ('R0', np.float_), ('gamma', np.float_),
            # Next event details.
            ('next_event', np.int_), ('next_time', np.float_),
        ]

    def can_smooth(self):
        """
        The fields that can be smoothed by the post-regularisation filter.
        """
        # Return the continuous model parameters.
        return {'R0', 'gamma'}

    def init(self, ctx, vec):
        """
        Initialise the state vectors.
        """
        # Initialise the model state variables.
        population = ctx.settings['model']['population_size']
        vec['S'] = population - 1
        vec['I'] = 1
        vec['R'] = 0

        # Initialise the model parameters.
        prior = ctx.data['prior']
        vec['R0'] = prior['R0']
        vec['gamma'] = prior['gamma']

        # Select the first event for event particle.
        vec['next_time'] = 0
        vec['next_event'] = 0
        self.select_next_event(ctx, vec, stop_time=0)

    def update(self, ctx, time_step, is_forecast, prev, curr):
        """
        Update the state vectors to account for all events that occur up to,
        and including, ``time``.
        """
        # Copy the state vectors, and update the current state.
        curr[:] = prev[:]

        # Simulate events for active particles.
        active = self.active_particles(curr, time_step.end)
        while any(active):
            # Simulate infection events.
            infections = np.logical_and(active, curr['next_event'] == 0)
            curr['S'][infections] -= 1
            curr['I'][infections] += 1

            # Simulate recovery events.
            recoveries = np.logical_and(active, curr['next_event'] == 1)
            curr['I'][recoveries] -= 1
            curr['R'][recoveries] += 1

            # Identifying which particles are still active.
            self.select_next_event(ctx, curr, stop_time=time_step.end)
            active = self.active_particles(curr, time_step.end)

    def active_particles(self, vec, stop_time):
        """
        Return a Boolean array that identifies the particles whose most recent
        event occurred no later than ``stop_time``.
        """
        return np.logical_and(
            vec['next_time'] <= stop_time,
            vec['I'] > 0,
        )

    def select_next_event(self, ctx, vec, stop_time):
        """
        Calculate the next event time and event type for each active particle.
        """
        active = self.active_particles(vec, stop_time)
        if not any(active):
            return

        # Extract state variables and parameters for the active particles.
        S = vec['S'][active]
        I = vec['I'][active]
        R0 = vec['R0'][active]
        gamma = vec['gamma'][active]
        N = ctx.settings['model']['population_size']

        # Calculate the mean rate of infection and recovery events.
        s_to_i_rate = R0 * gamma * S * I / (N - 1)
        i_to_r_rate = gamma * I
        rate_sum = s_to_i_rate + i_to_r_rate

        # Select the time of the next event.
        rng = ctx.component['random']['model']
        dt = - np.log(rng.random(S.shape)) / rate_sum
        vec['next_time'][active] += dt

        # Select the event type: False for infection and True for recovery.
        threshold = rng.random(S.shape) * rate_sum
        recovery_event = threshold > s_to_i_rate
        vec['next_event'][active] = recovery_event.astype(np.int_)


class SirDtmc(Model):
    """
    A discrete-time Markov chain implementation of the SIR model.

    The model settings must include the following keys:

    * ``population_size``: The number of individuals in the population.
    """

    def field_types(self, ctx):
        """
        Define the state vector structure.
        """
        return [
            # Model state variables.
            ('S', np.int_), ('I', np.int_), ('R', np.int_),
            # Model parameters.
            ('R0', np.float_), ('gamma', np.float_),
        ]

    def can_smooth(self):
        """
        The fields that can be smoothed by the post-regularisation filter.
        """
        # Return the continuous model parameters.
        return {'R0', 'gamma'}

    def init(self, ctx, vec):
        """
        Initialise the state vectors.
        """
        # Initialise the model state variables.
        population = ctx.settings['model']['population_size']
        vec['S'] = population - 1
        vec['I'] = 1
        vec['R'] = 0

        # Initialise the model parameters.
        prior = ctx.data['prior']
        vec['R0'] = prior['R0']
        vec['gamma'] = prior['gamma']

    def update(self, ctx, time_step, is_forecast, prev, curr):
        """
        Update the state vectors.
        """
        rng = ctx.component['random']['model']
        beta = prev['R0'] * prev['gamma']
        denom = ctx.settings['model']['population_size'] - 1

        # Calculate the rate at which *an individual* leaves S.
        s_out_rate = time_step.dt * beta * prev['I'] / denom
        # Select the number of infections.
        s_out = rng.binomial(prev['S'], - np.expm1(- s_out_rate))

        # Calculate the rate at which *an individual* leaves I.
        i_out_rate = time_step.dt * prev['gamma']
        # Select the number of recoveries.
        i_out = rng.binomial(prev['I'], - np.expm1(- i_out_rate))

        # Update the state variables.
        curr['S'] = prev['S'] - s_out
        curr['I'] = prev['I'] + s_out - i_out
        curr['R'] = prev['R'] + i_out

        # Copy the model parameters.
        curr['R0'] = prev['R0']
        curr['gamma'] = prev['gamma']


class SirOdeEuler(Model):
    """
    An ordinary differential equation implementation of the SIR model, which
    uses the forward Euler method.

    The model settings must include the following keys:

    * ``population_size``: The number of individuals in the population.
    """

    def field_types(self, ctx):
        """
        Define the state vector structure.
        """
        return [
            # Model state variables.
            ('S', np.float_), ('I', np.float_), ('R', np.float_),
            # Model parameters.
            ('R0', np.float_), ('gamma', np.float_),
        ]

    def can_smooth(self):
        """
        The fields that can be smoothed by the post-regularisation filter.
        """
        # Return the continuous model parameters.
        return {'R0', 'gamma'}

    def init(self, ctx, vec):
        """
        Initialise the state vectors.
        """
        # Initialise the model state variables.
        population = ctx.settings['model']['population_size']
        vec['S'] = population - 1
        vec['I'] = 1
        vec['R'] = 0

        # Initialise the model parameters.
        prior = ctx.data['prior']
        vec['R0'] = prior['R0']
        vec['gamma'] = prior['gamma']

    def update(self, ctx, time_step, is_forecast, prev, curr):
        """
        Update the state vectors.
        """
        # Calculate the flow rates out of S and I.
        beta = prev['R0'] * prev['gamma']
        N = ctx.settings['model']['population_size']
        s_out = time_step.dt * beta * prev['I'] * prev['S'] / N
        i_out = time_step.dt * prev['gamma'] * prev['I']

        # Update the state variables.
        curr['S'] = prev['S'] - s_out
        curr['I'] = prev['I'] + s_out - i_out
        curr['R'] = prev['R'] + i_out

        # Copy the model parameters.
        curr['R0'] = prev['R0']
        curr['gamma'] = prev['gamma']


class SirOdeRk(OdeModel):
    """
    An ordinary differential equation implementation of the SIR model, which
    uses the explicit Runge-Kutta method of order 5(4).

    The model settings must include the following keys:

    * ``population_size``: The number of individuals in the population.
    """

    def field_types(self, ctx):
        """
        Define the state vector structure.
        """
        return [
            # Model state variables.
            ('S', np.float_), ('I', np.float_), ('R', np.float_),
            # Model parameters.
            ('R0', np.float_), ('gamma', np.float_),
        ]

    def can_smooth(self):
        """
        The fields that can be smoothed by the post-regularisation filter.
        """
        # Return the continuous model parameters.
        return {'R0', 'gamma'}

    def init(self, ctx, vec):
        """
        Initialise the state vectors.
        """
        # Initialise the model state variables.
        self.population = ctx.settings['model']['population_size']
        vec['S'] = self.population - 1
        vec['I'] = 1
        vec['R'] = 0

        # Initialise the model parameters.
        prior = ctx.data['prior']
        vec['R0'] = prior['R0']
        vec['gamma'] = prior['gamma']

        # Define the integration method.
        self.method = 'RK45'

    def d_dt(self, time, xt, ctx, is_forecast):
        """
        The right-hand side of the system.
        """
        s_out = xt['R0'] * xt['gamma'] * xt['I'] * xt['S'] / self.population
        i_out = xt['gamma'] * xt['I']
        d_dt = np.zeros(xt.shape, dtype=xt.dtype)
        d_dt['S'] = - s_out
        d_dt['I'] = s_out - i_out
        d_dt['R'] = i_out
        return d_dt


class SirSde(Model):
    """
    A stochastic differential equation implementation of the SIR model.

    The model settings must include the following keys:

    * ``population_size``: The number of individuals in the population.
    """

    def field_types(self, ctx):
        """
        Define the state vector structure.
        """
        return [
            # Model state variables.
            ('S', np.float_), ('I', np.float_), ('R', np.float_),
            # Model parameters.
            ('R0', np.float_), ('gamma', np.float_),
        ]

    def can_smooth(self):
        """
        The fields that can be smoothed by the post-regularisation filter.
        """
        # Return the continuous model parameters.
        return {'R0', 'gamma'}

    def init(self, ctx, vec):
        """
        Initialise the state vectors.
        """
        # Initialise the model state variables.
        population = ctx.settings['model']['population_size']
        vec['S'] = population - 1
        vec['I'] = 1
        vec['R'] = 0

        # Initialise the model parameters.
        prior = ctx.data['prior']
        vec['R0'] = prior['R0']
        vec['gamma'] = prior['gamma']

    def update(self, ctx, time_step, is_forecast, prev, curr):
        """
        Update the state vectors.
        """
        rng = ctx.component['random']['model']
        beta = prev['R0'] * prev['gamma']
        N = ctx.settings['model']['population_size']
        size = prev.shape

        # Calculate the mean flows out of S and I.
        s_mean = time_step.dt * beta * prev['I'] * prev['S'] / N
        i_mean = time_step.dt * prev['gamma'] * prev['I']
        # Sample the stochastic term for each flow.
        s_stoch = np.sqrt(s_mean) * rng.normal(size=size)
        i_stoch = np.sqrt(i_mean) * rng.normal(size=size)
        # Calculate the stochastic flows out of S and I, ensuring that all
        # compartments remain non-negative.
        s_out = np.clip(s_mean + s_stoch, a_min=0, a_max=prev['S'])
        i_out = np.clip(i_mean + i_stoch, a_min=0, a_max=prev['I'])

        # Update the state variables.
        curr['S'] = prev['S'] - s_out
        curr['I'] = prev['I'] + s_out - i_out
        curr['R'] = prev['R'] + i_out

        # Copy the model parameters.
        curr['R0'] = prev['R0']
        curr['gamma'] = prev['gamma']


class SirObs(Univariate):
    r"""
    A binomial observation model for the example SIR models.

    .. math::

       \mathcal{L}(y_t \mid x_t) &\sim B(n, p)

       n &= S(t-\Delta) - S(t)

    :param obs_unit: A descriptive name for the data.
    :param settings: The observation model settings dictionary.

    The settings dictionary should contain the following keys:

    * ``observation_period``: The observation period :math:`\Delta`.

    For example, for daily observations that capture 80% of new infections:

    .. code-block:: toml

       [observations.cases]
       model = "pypfilt.examples.sir.SirObs"
       observation_period = 1
       parameters.p = 0.8
    """

    def new_infections(self, ctx, snapshot):
        r"""
        Return the number of new infections :math:`S(t-\Delta) - S(t)` that
        occurred during the observation period :math:`\Delta` for each
        particle.
        """
        period = self.settings['observation_period']
        prev = snapshot.back_n_units_state_vec(period)
        new_infs = prev['S'] - snapshot.state_vec['S']
        # Round continuous values to the nearest integer.
        if not np.issubdtype(new_infs.dtype, np.int_):
            new_infs = new_infs.round().astype(np.int_)
        if np.any(new_infs < 0):
            raise ValueError('Negative number of new infections')
        return new_infs

    def distribution(self, ctx, snapshot):
        """
        Return the observation distribution for each particle.
        """
        prob = ctx.settings['observations'][self.unit]['parameters']['p']
        infections = self.new_infections(ctx, snapshot)
        return scipy.stats.binom(n=infections, p=prob)
