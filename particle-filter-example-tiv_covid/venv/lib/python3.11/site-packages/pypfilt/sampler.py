"""Construct sampling functions for each model parameter."""

import abc
import lhs
import logging
import numpy as np

from .build import lookup


class Base(abc.ABC):
    """
    The base class for parameter samplers.
    """

    @abc.abstractmethod
    def draw_samples(self, settings, prng, particles, prior, sampled):
        """
        Return samples from the model prior distribution.

        :param settings: A dictionary of sampler-specific settings.
        :param prng: The source of randomness for the sampler.
        :param particles: The number of particles.
        :param prior: The prior distribution table.
        :param sampled: Sampled values for specific parameters.
        """
        pass


def sample_from(prng, particles, fn_name, args):
    """
    Return a function that draws samples from the specified distribution.

    :param fn_name: The name of a ``numpy.random.Generator`` method used to
        generate samples.
    :param args: A dictionary of keyword arguments.

    As a special case, ``fn_name`` may be set to ``'inverse_uniform'`` to
    sample from a uniform distribution and then take the reciprocal:

    .. math:: X \\sim \\frac{1}{\\mathcal{U}(a, b)}

    The bounds ``a`` and ``b`` may be specified by the following keyword
    arguments:

    + :code:`a = args['low']` **or** :code:`a = 1 / args['inv_low']`
    + :code:`b = args['high']` **or** :code:`b = 1 / args['inv_high']`
    """
    if fn_name == 'inverse_uniform':
        if 'low' in args:
            low = args['low']
        else:
            low = 1 / args['inv_low']

        if 'high' in args:
            high = args['high']
        else:
            high = 1 / args['inv_high']

        if low > high:
            # Return consistent outputs for NumPy < 1.21 and NumPy >= 1.21.
            # See https://github.com/numpy/numpy/pull/17921 for details.
            return -1 / prng.uniform(low=-low, high=-high, size=particles)

        return 1 / prng.uniform(low=low, high=high, size=particles)
    else:
        return getattr(prng, fn_name)(**args, size=particles)


def validate_prior(prior) -> None:
    """
    Ensure each prior comprises a function name and arguments dictionary,
    otherwise raise a ValueError.

    Note that this doesn't enforce that each prior corresponds to a known
    model parameter, because this would prevent us from supporting prior
    distributions that are expressed in terms of **transformed** parameters
    (such as reciprocals of rate parameters).

    :param prior: The prior distribution table.
    """
    logger = logging.getLogger(__name__)
    for (name, info) in prior.items():
        if 'name' not in info:
            raise ValueError('Missing prior name for {}'.format(name))
        elif not isinstance(info['name'], str):
            raise ValueError('Invalid prior name for {}'.format(name))
        if 'args' not in info:
            raise ValueError('Missing prior arguments for {}'.format(name))
        elif not isinstance(info['args'], dict):
            raise ValueError('Invalid prior arguments for {}'.format(name))
        if len(info) != 2:
            extra_keys = [k for k in info if k not in ['name', 'args']]
            logger.warning('Extra prior keys for %s: %s', name, extra_keys)


class Independent(Base):
    """
    The default sampler, which draws independent samples for each model
    parameter.
    """

    def draw_samples(self, settings, prng, particles, prior, sampled):
        validate_prior(prior)
        values = {
            name: sample_from(prng, particles, dist['name'], dist['args'])
            for (name, dist) in prior.items()
        }
        for (name, samples) in sampled.items():
            if name in values:
                msg_fmt = 'Values provided for parameter {}'
                raise ValueError(msg_fmt.format(name))
            values[name] = samples
        return values


class LatinHypercube(Base):
    """
    Draw parameter samples using Latin hypercube sampling.
    """

    def draw_samples(self, settings, prng, particles, prior, sampled):
        """
        Return samples from the model prior distribution.

        :param settings: A dictionary of sampler-specific settings.
        :param prng: The source of randomness for the sampler.
        :param particles: The number of particles.
        :param prior: The prior distribution table.
        :param sampled: Sampled values for specific parameters.
        """
        # Separate the independent and dependent parameters.
        indep_params = {
            name: dist for (name, dist) in prior.items()
            if not dist.get('dependent', False)
        }
        dep_params = {
            name: dist for (name, dist) in prior.items()
            if dist.get('dependent', False)
        }

        # Convert list arguments into numpy arrays.
        for table in [indep_params, dep_params]:
            for dist in table.values():
                args = dist.get('args', {})
                arg_names = list(args.keys())
                for arg_name in arg_names:
                    if isinstance(args[arg_name], list):
                        args[arg_name] = np.array(args[arg_name])

        dep_fn = settings.get('dependent_distributions_function')
        if dep_fn is None:
            if dep_params:
                raise ValueError('dependent_distributions_function missing')
            dep_params = None
        elif isinstance(dep_fn, str):
            # Turn a function name into the corresponding function.
            dep_fn = lookup(dep_fn)

        return lhs.draw(prng, particles, indep_params,
                        dep_params=dep_params, dep_fn=dep_fn,
                        values=sampled)
