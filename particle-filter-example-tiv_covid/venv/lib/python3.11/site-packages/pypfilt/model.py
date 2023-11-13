"""Base class for simulation models."""

import abc
import copy
import functools
import numpy as np
import operator
import scipy.integrate
import warnings


class Model(abc.ABC):
    """
    The base class for simulation models, which defines the minimal set of
    methods that are required.
    """

    def init(self, ctx, vec):
        """
        Initialise a matrix of state vectors.

        :param ctx: The simulation context.
        :param vec: An uninitialised :math:`P \\times S` matrix of state
            vectors, for :math:`P` particles and state vectors of length
            :math:`S` (as defined by :py:func:`~field_types`).

        .. note:: The default implementation initialises each state vector
           field to the prior samples of the same name, and produces warnings
           if there were unused prior samples.

           Models should only implement this method if the state vectors
           should not be initialised in this manner, or if there are other
           initialisation actions that must be performed.
        """
        for field in vec.dtype.names:
            vec[field] = ctx.data['prior'][field]
        unused_samples = set(ctx.data['prior']) - set(vec.dtype.names)
        for name in unused_samples:
            msg = 'Unused prior samples for "{}"'.format(name)
            warnings.warn(msg, UserWarning)

    @abc.abstractmethod
    def field_types(self, ctx):
        """
        Return a list of ``(field_name, field_dtype, field_shape)`` tuples
        that define the state vector.

        The third element, ``field_shape``, is optional and contains the shape
        of this field if it forms an array of type ``field_dtype``.

        These tuples **must** be in the same order as the state vector itself.

        :param ctx: The simulation context.
        """
        pass

    def field_names(self, ctx):
        """
        Return a list of the fields that define the state vector.

        These tuples **must** be in the same order as the state vector itself.

        :param ctx: The simulation context.
        """
        return [field[0] for field in self.field_types(ctx)]

    def can_smooth(self):
        """
        Return the set of field names in the state vector that can be smoothed
        by the post-regularised particle filter (see
        :func:`~pypfilt.resample.post_regularise`).

        .. note:: Models should only implement this method if the state vector
           contains parameters that can be smoothed.
        """
        return {}

    @abc.abstractmethod
    def update(self, ctx, time_step, is_fs, prev, curr):
        """
        Perform a single time-step, jumping forward to ``time`` and recording
        the updated particle states in ``curr``.

        :param ctx: The simulation context.
        :type ctx: ~pypfilt.build.Context
        :param time_step: The time-step details.
        :type time_step: ~pypfilt.pfilter.TimeStep
        :param is_fs: Indicates whether this is a forecasting simulation.
        :param prev: The state before the time-step.
        :param curr: The state after the time-step (destructively updated).
        """
        pass

    def resume_from_cache(self, ctx):
        """
        Notify the model that a simulation will begin from a saved state.

        The model does not need to initialise the state vectors, since these
        will have been loaded from a cache file, but it may need to update any
        internal variables (i.e., those not stored in the state vectors).

        .. note:: Models should only implement this method if they need to
           prepare for the simulation.
        """
        pass

    def stat_info(self):
        """
        Describe each statistic that can be calculated by this model as a
        ``(name, stat_fn)`` tuple, where ``name`` is a string that identifies
        the statistic and ``stat_fn`` is a function that calculates the value
        of the statistic.

        .. note:: Models should only implement this method if they define one
           or more statistics.
        """
        return []

    def is_valid(self, hist):
        """
        Identify particles whose state and parameters can be inspected. By
        default, this function returns ``True`` for all particles. Override
        this function to ensure that inchoate particles are correctly
        ignored.

        .. note:: Models should only implement this method if there are
           conditions where some particles should be ignored.
        """
        return np.ones(hist.shape, dtype=bool)


class OdeModel(Model):
    """
    A base class for systems of ordinary differential equations.

    The default integration method is the explicit Runge-Kutta method of order
    5(4) ("RK45").
    Sub-classes can override this by setting ``self.method`` to any
    supported integration method, and scenarios can override this by defining
    the ``"model.ode_method"`` setting.

    .. note:: Sub-classes **must** implement :meth:`~OdeModel.d_dt` (see
       below), and **must not** implement :meth:`~Model.update`.

    .. warning:: This only supports models whose state vectors contain
       floating-point fields (which may be scalar or multi-dimensional).
    """

    @abc.abstractmethod
    def d_dt(self, time, xt, ctx, is_forecast):
        """
        The right-hand side of the ODE system.

        :param time: The current (scalar) time.
        :param xt: The particle state vectors.
        :param ctx: The simulation context.
        :param is_forecast: True if this is a forecasting simulation.
        :returns: The rate of change for each field in the particle state
            vectors.
        """
        pass

    def _num_fields(self, state_vec):
        """
        Determine the number of (floating-point) values that are contained in
        each particle state vector, allowing for individual fields to be
        scalar or multi-dimensional.

        :param state_vec: An array of particle state vectors.
        :raises ValueError: if there are nested fields or non-float fields.
        """
        num_fields = 0
        for (dtype, offset) in state_vec.dtype.fields.values():
            if not np.issubdtype(dtype.base, float):
                raise ValueError(f'Unsupported field type: {dtype.base}')
            if len(dtype.shape) == 0:
                num_fields += 1
            elif len(dtype.shape) == 1:
                num_fields += dtype.shape[0]
            else:
                num_fields += functools.reduce(operator.mul, dtype.shape, 1)
        return num_fields

    def _time_range(self, ctx, time_step):
        """
        Determine the time range over which to integrate.

        :param ctx: The simulation context.
        :param time_step: The time-step details.
        """
        time_scale = ctx.component['time']
        t0 = time_scale.to_scalar(time_step.start)
        t1 = time_scale.to_scalar(time_step.end)
        return (t0, t1)

    def _d_dt(self, time, xt, ctx, is_forecast, dtype, num_fields):
        """
        A wrapper around the right-hand side function that reshapes the input
        and output arrays.
        """
        x_statevec = xt.view(dtype)
        rhs = self.d_dt(time, x_statevec, ctx, is_forecast)
        return rhs.view((np.float_, num_fields)).reshape(-1)

    def update(self, ctx, time_step, is_forecast, prev, curr):
        num_fields = self._num_fields(prev)
        time_range = self._time_range(ctx, time_step)
        default_method = getattr(self, 'ode_method', 'RK45')
        method = ctx.get_setting(['model', 'ode_method'], default_method)
        prev_xt = prev.view((np.float_, num_fields)).reshape(-1)
        result = scipy.integrate.solve_ivp(
            self._d_dt,
            time_range,
            prev_xt,
            args=(ctx, is_forecast, prev.dtype, num_fields),
            method=method)
        curr_xt = result.y[:, -1]
        curr[:] = curr_xt.view(prev.dtype)


def ministeps(mini_steps=None):
    """
    Wrap a model's ``update()`` method to perform multiple "mini-steps" for
    each time-step.

    :param mini_steps: The (optional) number of "mini-steps" to perform for
        each time-step.
        This can be overridden by providing a value for the
        ``"time.mini_steps_per_step"`` setting.
    """
    def decorator(update_method):
        num_setting = ['time', 'mini_steps_per_step']
        setting_name = '.'.join(num_setting)

        @functools.wraps(update_method)
        def wrapper(self, ctx, time_step, is_fs, prev, curr):
            # Determine the number of mini-steps to perform.
            mini_num = ctx.get_setting(num_setting, mini_steps)
            if mini_num is None:
                msg_fmt = 'Must define setting "{}"'
                raise ValueError(msg_fmt.format(setting_name))

            # Performing one mini-step per time-step is simple.
            if mini_num == 1:
                update_method(self, ctx, time_step, is_fs, prev, curr)
                return

            # Define the simulation period and time-step size.
            full_scale = ctx.component['time']
            mini_scale = copy.copy(full_scale)
            mini_per_unit = mini_num * ctx.settings['time']['steps_per_unit']
            mini_scale.set_period(time_step.start, time_step.end,
                                  mini_per_unit)

            # Create temporary arrays for the previous and current state.
            mini_prev = prev.copy()
            mini_curr = curr.copy()

            # Note that we need to substitute the time scale component.
            # This ensures that if the model uses any time methods, such as
            # to_scalar(), it will be consistent with the mini-step scale.
            ctx.component['time'] = mini_scale
            # Simulate each mini-step.
            for (mini_step_num, mini_step) in mini_scale.steps():
                update_method(self, ctx, mini_step, is_fs,
                              mini_prev, mini_curr)
                mini_prev, mini_curr = mini_curr, mini_prev
            # Restore the original time scale component.
            ctx.component['time'] = full_scale

            # Update the output state vectors.
            # NOTE: the final state is recorded in mini_prev, because we
            # switch mini_prev and mini_curr after each mini-step.
            curr[:] = mini_prev[:]

        return wrapper

    return decorator
