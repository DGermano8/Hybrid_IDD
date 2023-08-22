import scipy.stats
import numpy as np
from pypfilt.model import Model
from pypfilt.obs import Univariate, Obs
import pdb

# --------------------------------------------------------------------
# Define the process models
#
# - SIS_ODE :: ODE
# - SIS_CTMC :: CTMC
# - SIS_Hybrid :: Hybrid
# --------------------------------------------------------------------

class SIS_ODE(Model):
    """
    """

    def field_types(self, ctx):
        """
        """
        return [('S', np.float_),
                ('I', np.float_),
                ('N', np.float_),
                ('betaCoef', np.float_),
                ('gammaCoef', np.float_)]

    def init(self, ctx, vec):
        """
        """
        prior = ctx.data['prior']
        for p_ix in range(ctx.settings['num_replicates']):
            vec['S'][p_ix] = prior['S'][p_ix]
            vec['I'][p_ix] = prior['I'][p_ix]
            vec['N'][p_ix] = prior['S'][p_ix] + prior['I'][p_ix]
            vec['betaCoef'][p_ix] = prior['betaCoef'][p_ix]
            vec['gammaCoef'][p_ix] = prior['gammaCoef'][p_ix]

    def update(self, ctx, time_step, is_forecast, prev, curr):
        """
        (Destructively) update the state vector `curr`.
        """
        deriv = np.zeros(curr['betaCoef'].shape)
        for p_ix in range(ctx.settings['num_replicates']):
            curr['betaCoef'][p_ix] = prev['betaCoef'][p_ix]
            curr['gammaCoef'][p_ix] = prev['gammaCoef'][p_ix]
            deriv = (prev['betaCoef'][p_ix] * prev['S'][p_ix] * prev['I'][p_ix] / prev['N'][p_ix] - prev['gammaCoef'][p_ix] * prev['I'][p_ix])
            curr['S'][p_ix] = prev['S'][p_ix] - time_step.dt * deriv
            curr['I'][p_ix] = prev['I'][p_ix] + time_step.dt * deriv
            curr['N'][p_ix] = prev['N'][p_ix]


class SIS_CTMC(Model):
    """
    """
    def field_types(self, ctx):
        """
        """
        return [('S', np.int_),
                ('I', np.int_),
                ('N', np.int_),
                ('betaCoef', np.float_),
                ('gammaCoef', np.float_),
                ('next_event', np.int_),
                ('next_time', np.float_)]

    def init(self, ctx, vec):
        """
        """
        prior = ctx.data['prior']
        for p_ix in range(ctx.settings['num_replicates']):
            vec['S'][p_ix] = prior['S'][p_ix]
            vec['I'][p_ix] = prior['I'][p_ix]
            vec['N'][p_ix] = prior['S'][p_ix] + prior['I'][p_ix]
            vec['betaCoef'][p_ix] = prior['betaCoef'][p_ix]
            vec['gammaCoef'][p_ix] = prior['gammaCoef'][p_ix]
            vec['next_time'][p_ix] = 0
            vec['next_event'][p_ix] = 0
            self.select_next_event(ctx, vec[p_ix], stop_time=0)

    def update(self, ctx, time_step, is_forecast, prev, curr):
        """
        """
        for p_ix in range(ctx.settings['num_replicates']):
            curr_ptcl = prev[p_ix]
            while ((curr_ptcl['next_time'] <= time_step.end) and
                   (curr_ptcl['I'] > 0)):
                if curr_ptcl['next_event'] == 0:
                    curr_ptcl['S'] += 1
                    curr_ptcl['I'] -= 1
                elif curr_ptcl['next_event'] == 1:
                    curr_ptcl['S'] -= 1
                    curr_ptcl['I'] += 1
                else:
                    raise ValueError('Invalid event')
                self.select_next_event(ctx, curr_ptcl,
                                       stop_time=time_step.end)
            curr[p_ix] = curr_ptcl

    def select_next_event(self, ctx, ptcl, stop_time):
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

# --------------------------------------------------------------------
# Define the observation models
# --------------------------------------------------------------------

class UniformObservation(Univariate):
    """
    Observation without error.
    """
    def distribution(self, ctx, snapshot):
        expected_value = snapshot.state_vec['I']
        return scipy.stats.randint(low=np.round(expected_value),
                                   high=np.round(expected_value+1))

class GaussianStateObservation(Univariate):
    """
    Observation with Gaussian noise.
    """
    def distribution(self, ctx, snapshot):
        expected_value = snapshot.state_vec['I']
        return scipy.stats.norm(loc=expected_value, scale=0.01)

class NoisyStateObservation(Univariate):
    """
    Observation with Poisson noise.
    """
    def distribution(self, ctx, snapshot):
        expected_value = snapshot.state_vec['I']
        return scipy.stats.poisson(mu=expected_value)
