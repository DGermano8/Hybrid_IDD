"""Plotting utilities built on top of matplotlib."""

import abc
import contextlib
import itertools
import numpy as np


def _label(label_fmt, value, fac_name=None):
    """Return the appropriate label for a factor.

    :param label_fmt: The format string, or a function that returns the label.
    :param value: The value of the faceting variable.
    :param fac_name: The (optional) name of the faceting variable, used in
        error messages.
    :raises ValueError: if ``label_fmt`` is not a format string or a function.
    """
    if hasattr(label_fmt, 'format'):
        return label_fmt.format(value)
    elif callable(label_fmt):
        return label_fmt(value)
    else:
        if fac_name is None:
            msg = "invalid `label_fmt`: {}".format(label_fmt)
        else:
            msg = "invalid `label_fmt` for {}: {}".format(fac_name, label_fmt)
        raise ValueError(msg)


class Plot(abc.ABC):
    """
    The base class for plots that comprise multiple subplots.

    :param \\**kwargs: Extra arguments to pass to
        `pyplot.subplots <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.subplots>`__.
    :ivar fig: The :py:class:`matplotlib.figure.Figure` instance for the plot.
    :ivar axs: The :math:`M \\times N` array of :class:`matplotlib.axes.Axes`
        instances for each of the sub-plots (:math:`M` rows and :math:`N`
        columns).
    """

    def __init__(self, **kwargs):
        import matplotlib.pyplot as plt
        self.__hs = []
        self.__ls = []
        self.__lset = set()
        self.fig, self.axs = plt.subplots(**kwargs)

    @abc.abstractmethod
    def subplots(self):
        """
        Return an iterator that yields ``(axes, data)`` tuples for each
        subplot.
        """
        pass

    def add_to_legend(self, objs, replace=False):
        """
        Add plot objects to the list of items to show in the figure legend.

        :param replace: Whether to ignore objects which share a label with any
            object already in this list (default) or to replace such objects
            (set to ``True``).
        """
        for obj in objs:
            lbl = obj.get_label()
            if lbl in self.__lset:
                if replace:
                    self.__hs[self.__ls.index(lbl)] = obj
            else:
                self.__hs.append(obj)
                self.__ls.append(lbl)
                self.__lset.add(lbl)

    def legend(self, **kwargs):
        """
        Add a figure legend that lists the objects registered with
        :func:`~Plot.add_to_legend`.

        :param \\**kwargs: Extra arguments to pass to
            `Figure.legend <http://matplotlib.org/api/figure_api.html#matplotlib.figure.Figure.legend>`__.
        """
        self.fig.legend(self.__hs, self.__ls, **kwargs)

    def set_xlabel(self, text, dy, **kwargs):
        """
        Add an x-axis label that is centred across all subplots.

        :param text: The label text.
        :param dy: The vertical position of the label.
        :param \\**kwargs: Extra arguments to pass to
            `Figure.text <http://matplotlib.org/api/figure_api.html#matplotlib.figure.Figure.text>`__.
        """
        x0 = np.min([ax.get_position().x0 for ax in self.fig.get_axes()])
        x1 = np.max([ax.get_position().x1 for ax in self.fig.get_axes()])
        self.fig.text(0.5 * (x0 + x1), dy, text, ha='center', va='center',
                      **kwargs)

    def set_ylabel(self, text, dx, **kwargs):
        """
        Add an y-axis label that is centred across all subplots.

        :param text: The label text.
        :param dx: The horizontal position of the label.
        :param \\**kwargs: Extra arguments to pass to
            `Figure.text <http://matplotlib.org/api/figure_api.html#matplotlib.figure.Figure.text>`__.
        """
        y0 = np.min([ax.get_position().y0 for ax in self.fig.get_axes()])
        y1 = np.max([ax.get_position().y1 for ax in self.fig.get_axes()])
        self.fig.text(dx, 0.5 * (y0 + y1), text, ha='center', va='center',
                      rotation=90, **kwargs)

    def expand_x_lims(self, xs, pad_frac=0.05, pad_abs=None):
        """
        Increase the range of the x-axis, relative to the plot data.

        :param xs: The x-axis data.
        :param pad_frac: The fractional increase in range.
        :param pad_abs: The absolute increase in range.
        """
        xmin = np.min(xs)
        xmax = np.max(xs)
        xlim0 = xmin - (xmax - xmin) * pad_frac
        xlim1 = xmax + (xmax - xmin) * pad_frac
        if pad_abs is not None:
            xlim0 -= pad_abs
            xlim1 += pad_abs
        for ax in self.fig.get_axes():
            ax.set_xlim(xlim0, xlim1)

    def expand_y_lims(self, ys, pad_frac=0.05, pad_abs=None):
        """
        Increase the range of the y-axis, relative to the plot data.

        :param xs: The y-axis data.
        :param pad_frac: The fractional increase in range.
        :param pad_abs: The absolute increase in range.
        """
        ymin = np.min(ys)
        ymax = np.max(ys)
        ylim0 = ymin - (ymax - ymin) * pad_frac
        ylim1 = ymax + (ymax - ymin) * pad_frac
        if pad_abs is not None:
            ylim0 -= pad_abs
            ylim1 += pad_abs
        for ax in self.fig.get_axes():
            ax.set_ylim(ylim0, ylim1)

    def __date_locator(self, day, month, year):
        """
        Return a date tick locator for a specific number of days, months, or
        years.

        :param day: Locate ticks at every N days.
        :param month: Locate ticks at every N months.
        :param year: Locate ticks at every N years.

        :raises ValueError: unless exactly **one** of ``day``, ``month``, and
            ``year`` is not ``None``.
        """
        import matplotlib.dates as mdates

        if sum(x is not None for x in [day, month, year]) != 1:
            raise ValueError("Must specify one of: day, month, year")
        if day is not None:
            loc = mdates.DayLocator(interval=day)
        elif month is not None:
            loc = mdates.MonthLocator(interval=month)
        else:
            loc = mdates.YearLocator(base=year)
        return loc

    def scale_x_date(self, lbl_fmt, day=None, month=None, year=None):
        """
        Use a datetime scale to locate and label the x-axis ticks.

        :param lbl_fmt: The ``strftime()`` format string for tick labels.
        :param day: Locate ticks at every N days.
        :param month: Locate ticks at every N months.
        :param year: Locate ticks at every N years.

        :raises ValueError: unless exactly **one** of ``day``, ``month``, and
            ``year`` is specified.
        """
        import matplotlib.dates as mdates
        loc = self.__date_locator(day, month, year)

        for ax in self.fig.get_axes():
            ax.xaxis.set_major_formatter(mdates.DateFormatter(lbl_fmt))
            ax.xaxis.set_major_locator(loc)

    def scale_y_date(self, lbl_fmt, day=None, month=None, year=None):
        """
        Use a datetime scale to locate and label the y-axis ticks.

        :param lbl_fmt: The ``strftime()`` format string for tick labels.
        :param day: Locate ticks at every N days.
        :param month: Locate ticks at every N months.
        :param year: Locate ticks at every N years.

        :raises ValueError: unless exactly **one** of ``day``, ``month``, and
            ``year`` is specified.
        """
        import matplotlib.dates as mdates
        loc = self.__date_locator(day, month, year)

        for ax in self.fig.get_axes():
            ax.yaxis.set_major_formatter(mdates.DateFormatter(lbl_fmt))
            ax.yaxis.set_major_locator(loc)

    def save(self, filename, format, width, height, **kwargs):
        """
        Save the plot to disk (a thin wrapper for
        `savefig <http://matplotlib.org/api/figure_api.html#matplotlib.figure.Figure.savefig>`__).

        :param filename: The output filename or a Python file-like object.
        :param format: The output format.
        :param width: The figure width *in inches*.
        :param height: The figure height *in inches*.
        :param \\**kwargs: Extra arguments for ``savefig``; the defaults are
            ``transparent=True`` and ``bbox_inches='tight'``.
        """
        save_args = {'transparent': True, 'bbox_inches': 'tight'}
        save_args.update(kwargs)
        self.fig.set_figwidth(width)
        self.fig.set_figheight(height)
        self.fig.savefig(filename, format=format, **save_args)


class Single(Plot):
    """
    Faceted plots that contain only one sub-plot; i.e., a single plot that
    provides the same methods as faceted plots that contain many sub-plots.

    :param data: The NumPy array containing the data to plot.
    :param xlbl: The label for the x-axis.
    :param ylbl: The label for the y-axis.
    :param \\**kwargs: Extra arguments for :class:`~Plot`.
    """

    def __init__(self, data, xlbl, ylbl, **kwargs):
        self.__df = data
        self.__xlbl = xlbl
        self.__ylbl = ylbl
        # Ensure that self.axs is a 2-dimensional array (row x column).
        kwargs['squeeze'] = False
        super(Single, self).__init__(ncols=1, nrows=1, **kwargs)

    def expand_x_lims(self, col, pad_frac=0.05, pad_abs=None):
        """
        Increase the range of the x-axis, relative to the plot data.

        :param col: The column name for the x-axis data.
        :param pad_frac: The fractional increase in range.
        :param pad_abs: The absolute increase in range.
        """
        xs = self.__df[col]
        super(Single, self).expand_x_lims(xs, pad_frac, pad_abs)

    def expand_y_lims(self, col, pad_frac=0.05, pad_abs=None):
        """
        Increase the range of the y-axis, relative to the plot data.

        :param col: The column name for the y-axis data.
        :param pad_frac: The fractional increase in range.
        :param pad_abs: The absolute increase in range.
        """
        ys = self.__df[col]
        super(Single, self).expand_y_lims(ys, pad_frac, pad_abs)

    def subplots(self, hide_axes=False, dx=0.055, dy=0.025):
        """
        Return an iterator that yields ``(axes, data)`` tuples for each
        subplot.

        :param hide_axes: Whether to hide x and y axes that are not on their
            bottom or left edge, respectively, of the figure.
        :param dx: The horizontal location for the y-axis label.
        :param dy: The vertical location for the x-axis label.
        """
        self.set_xlabel(self.__xlbl, dy=dy)
        self.set_ylabel(self.__ylbl, dx=dx)
        yield self.axs[0, 0], self.__df


class Grid(Plot):
    """
    Faceted plots similar to those produced by ggplot2's ``facet_grid()``.

    :param data: The NumPy array containing the data to plot.
    :param xlbl: The label for the x-axis.
    :param ylbl: The label for the y-axis.
    :param xfac: The horizontal faceting variable, represented as a tuple
        ``(column_name, label_fmt)`` where ``column_name`` is the name of a
        column in ``data`` and ``label_fmt`` is the format string for facet
        labels or a function that returns the facet label.
    :param yfac: The vertical faceting variable (see ``xfac``).
    :param \\**kwargs: Extra arguments for :class:`~Plot`.
    """

    def __init__(self, data, xlbl, ylbl, xfac, yfac, **kwargs):
        self.__df = data
        self.__xlbl = xlbl
        self.__ylbl = ylbl
        self.__xfac = xfac
        self.__yfac = yfac
        self.__xfac_lvls = sorted(np.unique(data[xfac[0]]))
        self.__yfac_lvls = sorted(np.unique(data[yfac[0]]))
        nc, nr = len(self.__xfac_lvls), len(self.__yfac_lvls)
        # Ensure that self.axs is a 2-dimensional array (row x column).
        kwargs['squeeze'] = False
        super(Grid, self).__init__(ncols=nc, nrows=nr, **kwargs)

    def expand_x_lims(self, col, pad_frac=0.05, pad_abs=None):
        """
        Increase the range of the x-axis, relative to the plot data.

        :param col: The column name for the x-axis data.
        :param pad_frac: The fractional increase in range.
        :param pad_abs: The absolute increase in range.
        """
        xs = self.__df[col]
        super(Grid, self).expand_x_lims(xs, pad_frac, pad_abs)

    def expand_y_lims(self, col, pad_frac=0.05, pad_abs=None):
        """
        Increase the range of the y-axis, relative to the plot data.

        :param col: The column name for the y-axis data.
        :param pad_frac: The fractional increase in range.
        :param pad_abs: The absolute increase in range.
        """
        ys = self.__df[col]
        super(Grid, self).expand_y_lims(ys, pad_frac, pad_abs)

    def subplots(self, hide_axes=False, dx=0.055, dy=0.025):
        """
        Return an iterator that yields ``(axes, data)`` tuples for each
        subplot.

        :param hide_axes: Whether to hide x and y axes that are not on their
            bottom or left edge, respectively, of the figure.
        :param dx: The horizontal location for the y-axis label.
        :param dy: The vertical location for the x-axis label.
        """
        for rix, yf in enumerate(self.__yfac_lvls):
            for cix, xf in enumerate(self.__xfac_lvls):
                mask = np.logical_and(self.__df[self.__xfac[0]] == xf,
                                      self.__df[self.__yfac[0]] == yf)
                df = self.__df[mask]
                ax = self.axs[rix, cix]
                ax.yaxis.set_ticks_position('left')
                ax.xaxis.set_ticks_position('bottom')
                ax.label_outer()
                subplot = ax.get_subplotspec()
                if subplot.is_first_col():
                    ax.set_ylabel(self.__ylbl)
                if subplot.is_last_col():
                    ax2 = ax.twinx()
                    ax2.set_ylabel(_label(self.__yfac[1], yf, 'yfac'),
                                   rotation=270)
                    ax2.spines['right'].set_visible(False)
                    ax2.yaxis.set_ticks_position('none')
                    ax2.yaxis.set_ticklabels([])
                if subplot.is_last_row():
                    ax.set_xlabel(self.__xlbl)
                if subplot.is_first_row():
                    ax.set_title(_label(self.__xfac[1], xf, 'xfac'))
                ax.xaxis.get_label().set_alpha(0)
                ax.yaxis.get_label().set_alpha(0)
                if hide_axes:
                    if not subplot.is_first_col():
                        ax.spines['left'].set_visible(False)
                        ax.yaxis.set_ticks_position('none')
                    if not subplot.is_last_row():
                        ax.spines['bottom'].set_visible(False)
                        ax.xaxis.set_ticks_position('none')
                    if subplot.is_last_col():
                        ax2.spines['bottom'].set_visible(False)
                        ax2.spines['left'].set_visible(False)
                yield ax, df
        self.set_xlabel(self.__xlbl, dy=dy)
        self.set_ylabel(self.__ylbl, dx=dx)


class Wrap(Plot):
    """
    Faceted plots similar to those produced by ggplot2's ``facet_wrap()``.

    :param data: The NumPy array containing the data to plot.
    :param xlbl: The label for the x-axis.
    :param ylbl: The label for the y-axis.
    :param fac: The faceting variable, represented as a tuple
        ``(column_name, label_fmt)`` where ``column_name`` is the name of a
        column in ``data`` and ``label_fmt`` is the format string for facet
        labels or a function that returns the facet label.
    :param nr: The number of rows; **one** of ``nr`` and ``nc`` **must** be
        specified.
    :param nc: The number of columns; **one** of ``nr`` and ``nc`` **must** be
        specified.
    :param \\**kwargs: Extra arguments for :class:`~Plot`.

    :raises ValueError: if ``nr`` and ``nc`` are both ``None`` or are both
        specified.
    """

    def __init__(self, data, xlbl, ylbl, fac, nr=None, nc=None, **kwargs):
        self.__df = data
        self.__xlbl = xlbl
        self.__ylbl = ylbl
        self.__fac = fac
        self.__fac_lvls = sorted(np.unique(data[fac[0]]))
        self.__num_lvls = len(self.__fac_lvls)
        if nc is None and nr is None:
            raise ValueError("must specify one of 'nc', 'nr'")
        if nc is not None and nr is not None:
            raise ValueError("must specify one of 'nc', 'nr'")
        if nc is None:
            nc = np.ceil(self.__num_lvls / nr).astype(int)
        else:
            nr = np.ceil(self.__num_lvls / nc).astype(int)
        self.__num_rows = nr
        self.__num_cols = nc
        # Ensure that self.axs is a 2-dimensional array (row x column).
        kwargs['squeeze'] = False
        super(Wrap, self).__init__(ncols=nc, nrows=nr, **kwargs)

    def expand_x_lims(self, col, pad_frac=0.05, pad_abs=None):
        """
        Increase the range of the x-axis, relative to the plot data.

        :param col: The column name for the x-axis data.
        :param pad_frac: The fractional increase in range.
        :param pad_abs: The absolute increase in range.
        """
        xs = self.__df[col]
        super(Wrap, self).expand_x_lims(xs, pad_frac, pad_abs)

    def expand_y_lims(self, col, pad_frac=0.05, pad_abs=None):
        """
        Increase the range of the y-axis, relative to the plot data.

        :param col: The column name for the y-axis data.
        :param pad_frac: The fractional increase in range.
        :param pad_abs: The absolute increase in range.
        """
        ys = self.__df[col]
        super(Wrap, self).expand_y_lims(ys, pad_frac, pad_abs)

    def subplots(self, hide_axes=False, dx=0.055, dy=0.025):
        """
        Return an iterator that yields ``(axes, data)`` tuples for each
        subplot.

        :param hide_axes: Whether to hide x and y axes that are not on their
            bottom or left edge, respectively, of the figure.
        :param dx: The horizontal location for the y-axis label.
        :param dy: The vertical location for the x-axis label.
        """
        max_rix = self.__num_rows - 1
        extra_plots = self.__num_rows * self.__num_cols - self.__num_lvls
        extra_cix = self.__num_cols - extra_plots
        for rix in range(self.__num_rows):
            for cix in range(self.__num_cols):
                is_first_col = cix == 0
                is_last_row = rix == max_rix
                if extra_plots > 0:
                    is_last_vis_row = rix == max_rix - 1 and cix >= extra_cix
                else:
                    is_last_vis_row = False
                ax = self.axs[rix, cix]
                ix = self.__num_cols * rix + cix
                if ix >= self.__num_lvls:
                    # These axes may have been deleted previously.
                    if ax in self.fig.axes:
                        self.fig.delaxes(ax)
                    continue
                fac = self.__fac
                lvl = self.__fac_lvls[ix]
                df = self.__df[self.__df[fac[0]] == lvl]
                ax.yaxis.set_ticks_position('left')
                ax.xaxis.set_ticks_position('bottom')
                # Hide tick labels except on the outer-most sub-plots, but
                # handle the case where the last row isn't full.
                if not is_first_col:
                    ax.yaxis.set_tick_params(labelleft=False)
                if not is_last_row and not is_last_vis_row:
                    ax.xaxis.set_tick_params(labelbottom=False)
                if is_first_col:
                    ax.set_ylabel(self.__ylbl)
                if is_last_row:
                    ax.set_xlabel(self.__xlbl)
                ax.xaxis.get_label().set_alpha(0)
                ax.yaxis.get_label().set_alpha(0)
                ax.set_title(_label(fac[1], lvl, 'fac'))
                # Hide axes except on the outer-most sub-plots, but handle the
                # case where the last row isn't full.
                if hide_axes:
                    if not is_first_col:
                        ax.spines['left'].set_visible(False)
                        ax.yaxis.set_ticks_position('none')
                    if not is_last_row and not is_last_vis_row:
                        ax.spines['bottom'].set_visible(False)
                        ax.xaxis.set_ticks_position('none')
                yield ax, df
        self.set_xlabel(self.__xlbl, dy=dy)
        self.set_ylabel(self.__ylbl, dx=dx)


def default_style():
    """
    The style sheet provided by pypfilt.
    """
    text_colour = '#000000'
    axis_colour = '#000000'
    return {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Noto Sans', 'Open Sans'],
        # http://matplotlib.org/users/mathtext.html#mathtext-tutorial
        # 'mathtext.default': 'regular',
        'mathtext.fontset': 'stixsans',
        'legend.frameon': False,  # True if we want set background
        'legend.facecolor': 'white',  # Legend background colour
        'axes.spines.top': False,
        'axes.spines.right': False,
        'legend.fontsize': 'medium',
        'axes.titlesize': 'medium',
        'figure.titlesize': 'medium',
        'legend.edgecolor': 'white',
        'grid.alpha': 0,
        'text.color': text_colour,
        'axes.labelcolor': text_colour,
        'xtick.color': axis_colour,
        'ytick.color': axis_colour,
        'grid.color': axis_colour,
        'axes.edgecolor': axis_colour,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.width': 1,
        'ytick.major.width': 1,
    }


@contextlib.contextmanager
def apply_style(style=None):
    """
    Temporarily apply a style sheet.

    :param style: The style sheet to apply (default: :func:`default_style`).

    ::

       with apply_style():
           make_plots()
    """
    import matplotlib.pyplot as plt

    if style is None:
        style = default_style()

    with plt.style.context(style):
        yield


def n_colours(name, n):
    """
    Extract a fixed number of colours from a colour map.

    :param name: The colour map name (or a ``matplotlib.colors.Colormap``
        instance).
    :param n: The number of colours required.

    ::

        colours = n_colours('Blues', 3)
    """
    import matplotlib.cm as cm
    cmap = cm.get_cmap(name)
    return cmap(np.linspace(0, 1, n + 2))[1:-1]


def brewer_qual(name):
    """
    Qualitative palettes from the ColorBrewer project: ``'Accent'``,
    ``'Dark2'``, ``'Paired'``, ``'Pastel1'``, ``'Pastel2'``, ``'Set1'``,
    ``'Set2'``, ``'Set3'``.

    :raises ValueError: if the palette name is invalid.
    """
    colours = {
        'Accent': [
            '#7FC97F', '#BEAED4', '#FDC086', '#FFFF99', '#386CB0', '#F0027F',
            '#BF5B17', '#666666'],
        'Dark2': [
            '#1B9E77', '#D95F02', '#7570B3', '#E7298A', '#66A61E', '#E6AB02',
            '#A6761D', '#666666'],
        'Paired': [
            '#A6CEE3', '#1F78B4', '#B2DF8A', '#33A02C', '#FB9A99', '#E31A1C',
            '#FDBF6F', '#FF7F00', '#CAB2D6', '#6A3D9A', '#FFFF99', '#B15928'],
        'Pastel1': [
            '#FBB4AE', '#B3CDE3', '#CCEBC5', '#DECBE4', '#FED9A6', '#FFFFCC',
            '#E5D8BD', '#FDDAEC', '#F2F2F2'],
        'Pastel2': [
            '#B3E2CD', '#FDCDAC', '#CBD5E8', '#F4CAE4', '#E6F5C9', '#FFF2AE',
            '#F1E2CC', '#CCCCCC'],
        'Set1': [
            '#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00', '#FFFF33',
            '#A65628', '#F781BF', '#999999'],
        'Set2': [
            '#66C2A5', '#FC8D62', '#8DA0CB', '#E78AC3', '#A6D854', '#FFD92F',
            '#E5C494', '#B3B3B3'],
        'Set3': [
            '#8DD3C7', '#FFFFB3', '#BEBADA', '#FB8072', '#80B1D3', '#FDB462',
            '#B3DE69', '#FCCDE5', '#D9D9D9', '#BC80BD', '#CCEBC5', '#FFED6F'],
    }
    if name not in colours:
        raise ValueError("invalid palette: {}".format(name))
    else:
        return colours[name]


def colour_iter(col, palette, reverse=False):
    """
    Iterate over the unique (*sorted*) values in an array, returning a
    ``(value, colour)`` tuple for each of the values.

    :param col: The column of (unsorted, repeated) values.
    :param palette: The colour map name or a list of colours.
    :param reverse: Whether to sort the values in ascending (default) or
        descending order.
    """
    levels = sorted(np.unique(col), reverse=reverse)
    if isinstance(palette, list):
        n_pal = len(palette)
        n_lvl = len(levels)
        if n_pal >= n_lvl:
            colours = palette[:n_lvl]
        else:
            raise ValueError("Need {} colours, given {}".format(n_lvl, n_pal))
    else:
        colours = n_colours(palette, len(levels))
    return zip(levels, colours)


def cred_ints(ax, data, x, ci, palette='Blues', **kwargs):
    """
    Plot credible intervals as shaded regions.

    :param ax: The plot axes.
    :param data: The NumPy array containing the credible intervals.
    :param x: The name of the x-axis column.
    :param ci: The name of the credible interval column.
    :param palette: The colour map name or a list of colours.
    :param \\**kwargs: Extra arguments to pass to
        `Axes.plot <http://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.plot.html#matplotlib.axes.Axes.plot>`__
        and
        `Axes.fill_between <http://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.fill_between.html#matplotlib.axes.Axes.fill_between>`__.
    :returns: A list of the series that were plotted.
    """
    hs = []
    for ci, colour in colour_iter(data[ci], palette=palette, reverse=True):
        if ci > 0:
            label = '{:d}% CI'.format(ci)
        else:
            label = 'Median'
        df = data[data['prob'] == ci]
        y_min = df['ymin']
        y_max = df['ymax']
        args = {'edgecolor': 'None',
                'facecolor': colour}
        args.update(kwargs)
        ts = ax.fill_between(df[x], y_min, y_max, **args)
        ts.set_label(label)
        if ci == 0:
            args = {'linestyle': '-',
                    'linewidth': 1.5,
                    'solid_capstyle': 'butt',
                    'color': colour}
            args.update(kwargs)
            ln = ax.plot(df[x], y_min, **args)[0]
            ln.set_label("_{}".format(label))
        hs.append(ts)

    return hs


def observations(ax, data, label='Observations', future=False, **kwargs):
    """
    Plot observed values.

    :param ax: The plot axes.
    :param data: The NumPy array containing the observation data.
    :param label: The label for the observation data.
    :param future: Whether the observations occur after the forecasting time.
    :param \\**kwargs: Extra arguments to pass to
        `Axes.plot <http://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.plot.html#matplotlib.axes.Axes.plot>`__.
    :returns: A list of the series that were plotted.
    """
    if future:
        args = {'color': '#eee8d5',
                'edgecolors': 'black',
                'linewidths': 1}
    else:
        args = {'color': 'black'}
    args.update(kwargs)
    ts = ax.scatter(data['time'], data['value'], **args)
    ts.set_label(label)
    return [ts]


def series(ax, data, x, y, scales, legend_cols=True, **kwargs):
    """
    Add multiple series to a single plot, each of which is styled according to
    values in other columns.

    :param ax: The axes on which to draw the line series.
    :param data: The structured array that contains the data to plot.
    :param x: The name of the column that corresponds to the x-axis.
    :param y: The name of the column that corresponds to the y-axis.
    :param scales: A list of "scales" to apply to each line series; each scale
        is a tuple ``(column, kwarg, kwvals, label_fmt)`` where:

        - ``column`` is the name of a column in ``data``;
        - ``kwarg`` is the name of a keyword argument passed to ``plot()``;
        - ``kwvals`` is a list of values that the keyword argument will take;
          and
        - ``label_fmt`` is a format string for the legend keys or a function
          that returns the legend key.
    :param legend_cols: Whether to show each scale in a separate column.
    :param \\**kwargs: Extra arguments to pass to
        `Axes.plot <http://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.plot.html#matplotlib.axes.Axes.plot>`__.
    :returns: A list of the series that were plotted.

    ::

       scales = [
           # Colour lines according to the dispersion parameter.
           ('disp', 'color', brewer_qual('Set1'), r'$k = {:.0f}$'),
           # Vary line style according to the background signal.
           ('bg_obs', 'linestyle', ['-', '--', ':'], r'$bg_{{obs}} = {}$'),
       ]
       series(ax, data, 'x_col', 'y_col', scales)
    """
    import matplotlib.pyplot as plt

    cols = [scale[0] for scale in scales]
    names = [scale[1] for scale in scales]
    values = [scale[2] for scale in scales]
    fmts = [scale[3] for scale in scales]
    # Use np.sort to ensure `levels` is an array so that equality checks
    # return boolean arrays, even if the levels are strings.
    levels = [np.sort(np.unique(data[col])) for col in cols]
    rng = range(len(cols))

    # Plot each series.
    for vals in itertools.product(*levels):
        pairs = zip(cols, vals)
        masks = [data[col] == val for (col, val) in pairs]
        df = data[np.logical_and.reduce(tuple(masks))]
        df = df[np.argsort(df[x])]
        ixs = [np.where(levels[ix] == vals[ix])[0][0] for ix in rng]
        plot_args = {}
        plot_args.update(kwargs)
        plot_args.update({names[i]: values[i][ixs[i]] for i in rng})
        ax.plot(df[x], df[y], **plot_args)

    # Add each style as a separate (invisible) line, to act as a legend key.
    hs = []
    max_len = max(len(lvls) for lvls in levels)
    for ix, col in enumerate(cols):
        args = [{} for _ in levels[ix]]
        for lix, val in enumerate(levels[ix]):
            kw = args[lix]
            kw.update(kwargs)
            kw[names[ix]] = values[ix][lix]
            kw['label'] = _label(fmts[ix], levels[ix][lix], col)
            if names[ix] != 'color':
                kw['color'] = 'black'
        lines = [plt.Line2D([], [], **kw) for kw in args]
        # If we're showing each scale in a separate column, we must ensure
        # that each column contains the same number of items by adding blank
        # series where necessary.
        if legend_cols:
            while len(lines) < max_len:
                lines.append(plt.Line2D([], [], alpha=0, label=''))
        hs = hs + lines

    return hs
