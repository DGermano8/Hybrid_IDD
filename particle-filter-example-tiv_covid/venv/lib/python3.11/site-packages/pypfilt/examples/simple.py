import io
import numpy as np
import pkgutil
import pypfilt
import scipy


class GaussianWalk(pypfilt.Model):
    r"""
    A Gaussian random walk.

    .. math::

       x_t &= x_{t-1} + X_t \\
       X_t &\sim N(\mu = 0, \sigma = 1)

    The initial values :math:`x_0` are defined by the prior distribution for
    ``"x"``:

    .. code-block:: toml

       [prior]
       x = { name = "uniform", args.loc = 10.0, args.scale = 10.0 }
    """

    def field_types(self, ctx):
        return [('x', np.dtype(float))]

    def update(self, ctx, time_step, is_fs, prev, curr):
        """Perform a single time-step."""
        rnd = ctx.component['random']['model']
        step = rnd.normal(loc=0, scale=1, size=curr.shape)
        curr['x'] = prev['x'] + step


class GaussianObs(pypfilt.obs.Univariate):
    r"""
    A Gaussian observation model for the GaussianWalk model.

    .. math::

       \mathcal{L}(y_t \mid x_t) \sim N(\mu = x_t, \sigma = s)

    The observation model has one parameter: the standard deviation :math:`s`,
    whose value is defined by the ``"parameters.sdev"`` setting:

    .. code-block:: toml

       [observations.x]
       model = "pypfilt.examples.simple.GaussianObs"
       parameters.sdev = 0.2
    """
    def distribution(self, ctx, snapshot):
        expected = snapshot.state_vec['x']
        sdev = self.settings['parameters']['sdev']
        return scipy.stats.norm(loc=expected, scale=sdev)


def __example_data(filename):
    return pkgutil.get_data('pypfilt.examples', filename).decode()


def gaussian_walk_toml_data():
    """Return the contents of the example file "gaussian_walk.toml"."""
    return __example_data('gaussian_walk.toml')


def gaussian_walk_instance():
    """
    Return an instance of the simple example scenario.
    """
    toml_input = io.StringIO(gaussian_walk_toml_data())
    instances = list(pypfilt.scenario.load_instances(toml_input))
    assert len(instances) == 1
    return instances[0]
