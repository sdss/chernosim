#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2020-09-27
# @Filename: plotting.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import pathlib

import matplotlib
import numpy
import pandas
import seaborn


seaborn.set_theme(style='darkgrid', font='serif')


def _process_plot(ax, outfile=None, savefig_kws={}):
    """Saves and returns the plot."""

    if outfile:
        ax.figure.savefig(outfile, **savefig_kws)

    matplotlib.pyplot.close()

    return ax


def plot_rmse_hist(data, clip=[0, 5], **kwargs):
    """Plots the RMSE as a histogram."""

    ax = seaborn.histplot(data, x='rmse', binrange=clip,
                          kde=True, kde_kws={'clip': clip},
                          element='step')
    ax.set_xlabel('RMSE [arcsec]')

    return _process_plot(ax, **kwargs)


def plot_fwhm_rmse(data, window=500, **kwargs):
    """Plots FHWM vs RMSE smoothed with a rolling average."""

    f_r = data.loc[:, ['fwhm', 'rmse']]
    f_r = f_r.sort_values('fwhm').dropna().rolling(window=window,
                                                   min_periods=1).mean()

    ax = seaborn.lineplot(data=f_r, x='fwhm', y='rmse', ci=None)

    ax.set_xlabel('FWHM [arcsec]')
    ax.set_ylabel('RMSE [arcsec]')

    return _process_plot(ax, **kwargs)


def plot_solved(data, x, binwidth, clip=[0, 100], xlabel=None, **kwargs):
    """Plots a stacked histogram of solved/unsolved vs a variable."""

    clipped_data = data[(data[x] >= clip[0]) & (data[x] <= clip[1])].copy()

    clip[0] = clipped_data[x].min()
    clip[1] = clipped_data[x].max()

    n_bins = int((clip[1] - clip[0]) / binwidth)
    clipped_data.loc[:, 'bin'] = pandas.cut(clipped_data.loc[:, x], n_bins)

    bins_len = clipped_data.groupby('bin').apply(len)
    zero_idx = numpy.where(bins_len == 0)
    if len(zero_idx[0]) > 0:
        clip[1] = bins_len.index[zero_idx[0][0]].left

    # Get the number of measurements in each bin and the number of them that
    # were solved.
    binned = clipped_data.groupby('bin').apply(lambda x: (len(x),
                                                          x.solved.sum()))

    # Unpack the tuple and rename columns. Add a column with the left value
    # of the interval, which we'll use for the histogram.
    binned = binned.apply(pandas.Series)
    binned.columns = ['n', 'n_solved']
    binned.loc[:, 'bin_x'] = list(map(lambda x: x.left, binned.index))

    # Normalise each bin independently so that its total probability is 1.
    binned.loc[:, 'prob_solved'] = binned.n_solved / binned.n
    binned.loc[:, 'prob_not_solved'] = 1. - binned.prob_solved

    binned = binned.set_index('bin_x')
    binned.loc[clip[1], ['prob_solved',
                         'prob_not_solved']] = binned.iloc[-1][['prob_solved',
                                                                'prob_not_solved']]

    # Override colour palette to use red for unsolved and blue for solved.
    c_deep = seaborn.color_palette('deep')
    with seaborn.color_palette([c_deep[1], c_deep[0]]):

        _, ax = matplotlib.pyplot.subplots()

        ax.fill_between(binned.index, binned.prob_solved,
                        step='post', lw=0.0, alpha=0.8, label='Solved')
        ax.fill_between(binned.index,
                        binned.prob_solved + binned.prob_not_solved,
                        y2=binned.prob_solved,
                        step='post', lw=0.0, alpha=0.8, label='Not solved')

    ax.set_xlabel(xlabel or x)
    ax.set_ylabel('Probability')

    ax.set_xlim(clip)
    ax.set_ylim(0, 1)

    ax.legend(loc='upper left')

    return _process_plot(ax, **kwargs)


def plot_n_detected_solve_time(data, window=10, **kwargs):
    """Plots n_detected vs average solve time with a rolling average."""

    _, ax = matplotlib.pyplot.subplots()

    for solved in [True, False]:
        n_s = data.loc[data.solved == solved, ['n_detected', 'solve_time_avg']]
        solve_time = n_s.groupby('n_detected').apply(numpy.mean)['solve_time_avg']

        ax.scatter(solve_time.index, solve_time, s=30, alpha=0.4, lw=0.0,
                   color='g' if solved else 'r', zorder=20 if solved else 10)

        avg = solve_time.rolling(window=window, min_periods=1,
                                 win_type='triang').mean()

        ax.plot(avg.index, avg, color='g' if solved else 'r',
                zorder=20 if solved else 10,
                label='Solved' if solved else 'Not solved')

    x_max = data.n_detected.max()
    if x_max > 400:
        x_max = 400

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 20)

    ax.legend()

    ax.set_xlabel('n_detected')
    ax.set_ylabel('Average solve time [s]')

    return _process_plot(ax, **kwargs)


def plot_fields(data, org=120, **kwargs):
    """Plots the location of fields on a Mollweide projection."""

    data = data.groupby(['field', 'observatory']).first()

    fig = matplotlib.pyplot.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='mollweide')

    tick_labels = numpy.array([150., 120, 90, 60, 30, 0,
                               330, 300, 270, 240, 210])
    tick_labels = numpy.remainder(tick_labels + 360 + org, 360)
    tick_labels = numpy.array(tick_labels / 15., int)

    tick_str = []
    for tick_label in tick_labels[1::2]:
        tick_str.append('')
        tick_str.append('${0:d}^h$'.format(tick_label))
    tick_str.append('')

    ax.set_xticklabels(tick_str)

    ra = data.field_ra
    dec = data.field_dec

    import astropy.coordinates

    coord = astropy.coordinates.SkyCoord(ra=ra, dec=dec, unit='deg')
    gal = coord.galactic
    ra = gal.l.deg
    dec = gal.b.deg

    ra = numpy.remainder(ra + 360 - org, 360)  # shift RA values
    ind = ra > 180.
    ra[ind] -= 360  # scale conversion to [-180, 180]
    ra = -ra  # reverse the scale: East to the left

    color = list(map(lambda x: 'b' if x == 'apo' else 'r',
                     data.index.get_level_values('observatory')))

    ax.scatter(numpy.radians(ra), numpy.radians(dec),
               marker='x', color=color, s=1)

    ax.set_xlabel(r'$l_{2015.5}$')
    ax.set_ylabel(r'$b_{2015.5}$')

    return _process_plot(ax, **kwargs)


def generate_plots(path, format='png'):
    """Generate paths from a results file."""

    dirname = pathlib.Path(path).parent

    data = pandas.read_hdf(path)

    with matplotlib.rc_context(rc={'interactive': False}):

        plot_rmse_hist(data, outfile=dirname / ('rmse_hist' + f'.{format}'),
                       savefig_kws={'dpi': 300})

        plot_fwhm_rmse(data, outfile=dirname / ('fwhm_rmse' + f'.{format}'),
                       savefig_kws={'dpi': 300})

        plot_solved(data, 'n_detected', 1, clip=[0, 100],
                    savefig_kws={'dpi': 300},
                    outfile=dirname / ('n_detected_solved' + f'.{format}'))

        plot_solved(data, 'fwhm', 0.1, clip=[0.5, 3],
                    xlabel='FHWM [arcsec]', savefig_kws={'dpi': 300},
                    outfile=dirname / ('fwhm_solved' + f'.{format}'))

        plot_n_detected_solve_time(data, savefig_kws={'dpi': 300},
                                   outfile=dirname /
                                   ('n_detected_solve_time' + f'.{format}'))

        plot_fields(data, savefig_kws={'dpi': 300},
                    outfile=dirname / ('fields' + f'.{format}'))

        non_solved = data.groupby('field').filter(lambda x: sum(x.solved) == 0)
        plot_fields(non_solved, savefig_kws={'dpi': 300},
                    outfile=dirname / ('fields_non_solved' + f'.{format}'))
