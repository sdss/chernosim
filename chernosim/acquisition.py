#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2020-09-13
# @Filename: acquisition.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import functools
import multiprocessing
import pathlib
import shutil
import warnings

import astropy.table
import matplotlib.patches
import matplotlib.pyplot
import numpy
import yaml

from cherno.astrometry import AstrometryNet
from sdssdb.peewee.sdss5db import database
from sdsstools import read_yaml_file

from . import config
from .utils import (create_gfa_wcs, get_gfa_centre,
                    get_uniform_ra_dec, query_field, sky_separation)


def select_stars(data, boresight, observatory='apo',
                 r1=None, r2=None, phi=None, gfa_rot=None):
    """Selects stars for the simulation.

    Given a dataframe with a list of stars, returns a subset of the dataframe
    with stars that fall within the footprint of the GFA chips.

    The GFAs are defined as the areas that subtend an angle ``phi`` with
    respect to the boresight in an annulus of radii ``r1`` and ``r2``. The
    rotation angle of each camera is one of the ``gfa_rot`` values, with zero
    degrees corresponding to the direction of the celestial North. This is an
    approximation of the real footprint of the GFA, which are rectangular and
    not an annulus sector, but the areas are comparable and this provides a
    simple way to select the stars.

    Parameters
    ----------
    data : pandas.DataFrame
        A dataframe with the star data. Must contain at least two columns,
        ``ra`` and ``dec``, in degrees.
    boresight : tuple
        A tuple with the right ascension and declination of the boresight,
        in degrees.
    observatory : str
        The observatory, used to load the default configuration for the GFAs.
    r1,r2 : float
        The internal and external radii along which the GFAs are located, in
        degrees.
    phi : float
        The angle subtended by each GFA, in degrees.
    gfa_rot : list
        A list with the rotation of each GFA, with respect to the boresight,
        in degrees.

    Returns
    -------
    `~pandas.DataFrame`
        The input dataframe restricted to the stars that fall within the
        footprint of each GFA. A new column ``gfa`` is added with the index
        of the GFA, which correspond to the ``gfa_rot`` rotation.

    """

    # Get data from configuration file if not provided.
    obs_data = config[observatory]
    r1 = r1 or obs_data['r1']
    r2 = r2 or obs_data['r2']
    phi = phi or obs_data['phi']
    gfa_rot = gfa_rot or obs_data['gfa_rot']

    b_ra = boresight[0]
    b_dec = boresight[1]

    ra_rad = numpy.radians(data.ra)
    dec_rad = numpy.radians(data.dec)
    delta_ra_rad = ra_rad - numpy.radians(b_ra)

    # Calculate the separation between each star and the boresight.
    sep = numpy.degrees(
        numpy.arccos(
            numpy.sin(dec_rad) * numpy.sin(numpy.radians(b_dec)) +
            numpy.cos(dec_rad) * numpy.cos(numpy.radians(b_dec)) *
            numpy.cos(delta_ra_rad)
        )
    )

    # Remove stars that ar not in the GFA annulus
    data = data.loc[(sep > r1) & (sep < r2)]
    sep = sep[(sep > r1) & (sep < r2)]
    sep_rad = numpy.radians(sep)

    ra_rad = numpy.radians(data.ra)
    dec_rad = numpy.radians(data.dec)
    delta_ra_rad = ra_rad - numpy.radians(b_ra)

    # Calculate the angle, theta, between boresight, North, and the star.
    # We define a spherical triangle with vertices in North, boresight, and
    # each star and use the sine law.
    sin_theta = numpy.sin(delta_ra_rad) * numpy.cos(dec_rad) / numpy.sin(sep_rad)
    theta = numpy.degrees(numpy.arcsin(sin_theta))

    # Solve for degeneracy in arcsin.
    theta.loc[data.dec < b_dec] = 180 - theta[data.dec < b_dec]
    theta.loc[theta < 0] += 360

    data['theta'] = theta

    # Determine the GFA on which footprint each star falls, if any.
    data['gfa'] = -1

    for gfa_id in range(len(gfa_rot)):
        rot = gfa_rot[gfa_id]
        rot_min = (rot - phi / 2.) % 360.
        rot_max = (rot + phi / 2.) % 360.
        data.loc[(theta - rot_min) % 360. <=
                 (rot_max - rot_min) % 360., 'gfa'] = gfa_id

    data = data.loc[data.gfa >= 0]

    return data


def radec_to_xy(data, wcs=None, **kwargs):
    """Converts ``(RA, Dec)`` to ``(x, y)`` for a given GFA.

    Creates a mock WCS transformation for a given GFA and converts star RA, Dec
    to x, y on what would be a GFA image. This conversion is not carefully
    done and is not a proper transformation between on-sky coordinates and
    focal coordinates, but should be sufficient for the purposes of the
    simulation.

    Parameters
    ----------
    data : pandas.DataFrame
        A dataframe with the star data. Must contain at least two columns,
        ``ra`` and ``dec``, in degrees.
    wcs : ~astropy.wcs.WCS
        The WCS object to use. If `None`, it calls `.create_gfa_wcs`.
    kwargs : dict
        Arguments to pass to `.create_gfa_wcs`.

    Returns
    -------
    `~pandas.DataFrame`, `~astropy.wcs.WCS`
        The input dataframe with two columns, ``x`` and ``y`` indicating the
        position of the star on the GFA chip, and the `~astropy.wcs.WCS`
        object.

    """

    if len(data) == 0:
        data['x'] = numpy.nan
        data['y'] = numpy.nan
        return data

    if not wcs:
        wcs = create_gfa_wcs(**kwargs)

    # Convert coordinates to x, y
    coords = data[['ra', 'dec']].to_numpy()

    x, y = wcs.wcs_world2pix(coords, 0).T
    data['x'] = x
    data['y'] = y

    return data, wcs


def prepare_data(boresight, data=None, observatory='apo', r1=None, r2=None,
                 g_mag_range=None, phi=None, gfa_rot=None, shape=None,
                 pixel_size=None, plate_scale=None, plot=False,
                 apply_proper_motion=False, ref_epoch=2015.5, epoch=False):
    """Prepares data to be matched by astrometry.net.

    Performs the following steps:

    - Queries the database to receive the list of observed stars.

    - Applies proper motions.

    - Select stars that fall within the footprint of the GFAs, for a
      given footprint.

    - Calculates the WCS of each GFA and convert the input coordinates to
      pixel coordinates on the GFA image.

    - Creates a global WCS for the full FOV of the telescope, with zero on
      the boresight, and converts the input coordinates to pseudo-pixels in
      that frame.

    Parameters
    ----------
    boresight : tuple
        A tuple with the right ascension and declination of the boresight,
        in degrees.
    data : pandas.DataFrame
        A dataframe with the star data. Must contain at least two columns,
        ``ra`` and ``dec``, in degrees. If `None`, calls `.query_field` to
        retrieve a list of stars from the database.
    observatory : str
        The observatory, used to load the default configuration for the GFAs.
    r1,r2 : float
        The internal and external radii along which the GFAs are located, in
        degrees.
    g_mag_range : tuple
        The range of Gaia DR2 G magnitudes used to select stars.
    phi : float
        The angle subtended by each GFA, in degrees.
    gfa_rot : list
        A list with the rotation of each GFA, with respect to the boresight,
        in degrees.
    shape : tuple
        Number of pixels, in the x and y direction of the GFA chip.
    pixel_size : float
        The pixel size, in microns.
    plate_scale : float
        The plate scale, in mm/deg.
    plot : bool or str
        Whether to produce a plot with the input stars, GFA centres, and
        footprints. If a string, the path where to save the plot.
    apply_proper_motion : bool
        Whether to propagate the position to a given ``epoch``. Assumes the
        data returned by `.query_field` has columns ``pmra`` and ``pmdec`` in
        mas and that ``pmra`` contains a factor with the cosine of declination.
    ref_epoch : float
        The epoch of the catalogue, as a Julian year.
    epoch : float
        The epoch of the observation, as a Julian year.

    Returns
    -------
    `~pandas.DataFrame`
        The input dataframe restricted to the stars that fall within the
        footprint of each GFA and with additional column indicating the GFA
        chip and x and y positions on that chip, and the global x and y
        pixel coordinates on the pseudo-frame of the FOV.

    """

    b_ra, b_dec = boresight

    if data is None:
        data = query_field(boresight, r1=r1, r2=r2, observatory=observatory,
                           g_mag_range=g_mag_range)

    data = select_stars(data, boresight, observatory=observatory,
                        r1=r1, r2=r2, phi=phi, gfa_rot=gfa_rot)

    if apply_proper_motion:
        assert epoch is not None, 'epoch is needed to apply proper motions.'
        data['ra_orig'] = data.ra
        data['dec_orig'] = data.dec
        pmra = data.pmra / 1000 / 3600. / numpy.cos(numpy.radians(data.dec))
        pmdec = data.pmdec / 1000 / 3600.
        data.ra += pmra * (epoch - 2015.5)
        data.dec += pmdec * (epoch - 2015.5)

    obs_data = config[observatory]
    gfa_rot = gfa_rot or obs_data['gfa_rot']
    plate_scale = plate_scale or obs_data['plate_scale']
    pixel_size = pixel_size or config['gfa']['pixel_size']

    wcs = [create_gfa_wcs(rot,
                          boresight,
                          observatory='apo',
                          r1=r1, r2=r2,
                          shape=shape,
                          pixel_size=pixel_size,
                          plate_scale=plate_scale)
           for rot in gfa_rot]

    data = data.groupby('gfa').apply(
        lambda data_gfa: radec_to_xy(data_gfa,
                                     wcs=wcs[data_gfa.gfa.iloc[0]])[0])

    if plot is not False and plot is not None:

        fig, ax = matplotlib.pyplot.subplots()

        centres = numpy.array([get_gfa_centre(rot,
                                              boresight,
                                              observatory=observatory,
                                              r1=r1, r2=r2)
                               for rot in gfa_rot])

        ax.scatter(data.ra, data.dec, s=1.0, marker='.', color='b')
        ax.scatter(centres[:, 0], centres[:, 1], s=5.0, marker='x', color='r')

        obs_data = config[observatory]
        shape = shape or config['gfa']['shape']

        for ww in wcs:
            footprint = ww.calc_footprint(axes=shape)
            rect = matplotlib.patches.Polygon(footprint, facecolor='None',
                                              edgecolor='k', linewidth=1)
            ax.add_patch(rect)

        ax.set_xlim(b_ra + 1.6 / numpy.cos(numpy.radians(b_dec)),
                    b_ra - 1.6 / numpy.cos(numpy.radians(b_dec)))
        ax.set_ylim(b_dec - 1.6, b_dec + 1.6)

        n_stars = data.groupby('gfa').size().tolist()
        ax.set_title(f'(alpha, delta)=({b_ra:.2f}, {b_dec:.2f})\n '
                     f'n_stars={sum(n_stars)} '
                     f'({", ".join(map(str, n_stars))})')

        ax.set_xlabel('Right Ascension [deg]')
        ax.set_ylabel('Declination [deg]')

        fig.savefig(plot or 'gfa.pdf')

    return data


def add_noise(data, fwhm, detection_rate=0.95, non_detection_factor=1,
              mag_thres=13, mag_column='phot_g_mean_mag'):
    r"""Adds centroiding noise to the catalogue data.

    Modifies the pixel coordinates in the ``data`` dataframe, adding Gaussian
    noise with :math:`\sigma={\rm FWHM}/2\sqrt{2\ln 2}` to simulate seeing.
    If ``detection_rate`` is less than 1, targets are marked as detected or
    non-detected based on the ``dection_rate`` logarithmically scaled with
    magnitude.

    Parameters
    ----------
    data : pandas.DataFrame
        A dataframe with the star data. Must contain at least two columns,
        ``x`` and ``y``, with the pixel coordinates.
    fwhm : float
        The FWHM of the Gaussian noise to add. It must be in pixel units.
    detection_rate : float
        The probability of a source to be detected and its centroid measured.
    non_detection_factor : float
        A proportional factor used to weight the detection rate so that
        :math:`d=d_0-a\log(m-m_0)` where :math:`d` is the final detection rate
        that will be applied to a target, :math:`d_0` is initial
        ``detection_rate`, :math:`m` is the magnitude of the source,
        :math:`m_0` is ``mag_thres``, and :math:`a` is
        ``non_detection_factor``.
    mag_thres : float
        The magnitude above which the detection rate will be reduced
        logarithmically.
    mag_column : str
        The name of the magnitude column in the dataframe.

    Returns
    -------
    `~pandas.DataFrame`
        The input dataframe in which the pixel coordinates have been modified
        to add centroiding noise. An additional boolean column, ``detected``,
        is added to indicate whether the source has been detected following the
        logic described for ``detection_rate``.

    """

    sigma = fwhm / (2.0 * numpy.sqrt(2 * numpy.log(2)))

    n = data.shape[0]
    data.x += numpy.random.normal(0, sigma, n)
    data.y += numpy.random.normal(0, sigma, n)

    data['detected'] = True

    if detection_rate >= 1.0 or not mag_column:
        return data

    if not mag_thres:
        mag_thres = data[mag_column].max()

    delta_mag = data[mag_column] - mag_thres
    delta_mag[delta_mag < 0] = 0.

    detection_rate -= numpy.log10(delta_mag) * non_detection_factor

    non_detected = numpy.random.uniform(size=n) > detection_rate
    data.loc[:, 'detected'] = ~non_detected

    return data


def _do_one_field(fields, config_data, observatory, output_dir,
                  n_attempts, field_idx, database_profile=None):
    """Simulates one field."""

    if database_profile:
        database.set_profile(database_profile)

    boresight = fields[field_idx]
    field_id = field_idx + 1

    numpy.random.seed(config_data['seed'] + field_id)

    astrometry_cfg = config_data['astrometry.cfg']
    if not astrometry_cfg:
        astrometry_cfg = (pathlib.Path(__file__).parent.absolute() /
                          'etc/astrometry.cfg')

    field_dir = output_dir / f'{field_id:05d}'

    data = prepare_data(boresight, observatory=observatory,
                        g_mag_range=config_data['g_mag_range'],
                        apply_proper_motion=True, epoch=config_data['epoch'],
                        plot=False, shape=config_data['gfa']['shape'],
                        pixel_size=config_data['gfa']['pixel_size'],
                        **config_data[observatory])

    gfa_rot = config_data[observatory]['gfa_rot']
    gfa_centres = {gfa_id: get_gfa_centre(gfa_rot[gfa_id],
                                          boresight,
                                          observatory=observatory).tolist()
                   for gfa_id in range(len(gfa_rot))}

    for nn in range(n_attempts):

        n_att = nn + 1
        prefix = f'_{field_id:05d}_{n_att:03d}'

        log_config = {}
        log_config['simulation_config'] = config_data.copy()

        log_config['input'] = {}
        log_input = log_config['input']
        log_input['boresight'] = boresight.tolist()
        log_input['observatory'] = observatory
        log_input['field_id'] = field_id
        log_input['attempt_id'] = n_att
        log_input['gfa_centres'] = gfa_centres

        att_dir = field_dir / f'{n_att:03d}'
        if att_dir.exists():
            shutil.rmtree(att_dir)
        att_dir.mkdir(parents=True, exist_ok=True)

        fwhm = numpy.random.uniform(*config_data['fwhm_range'])
        log_input['fwhm'] = fwhm

        att_data = data.copy()
        att_data = add_noise(
            att_data, fwhm,
            detection_rate=config_data['detection_rate'],
            non_detection_factor=config_data['non_detection_factor'],
            mag_thres=config_data['mag_thres'],
            mag_column=config_data['mag_column'])

        log_input['n_stars'] = len(att_data)
        log_input['n_detected'] = len(att_data[att_data.detected])

        gfa_ids = range(config_data['gfa']['n_cameras'])
        log_input['n_stars_per_gfa'] = {i: 0 for i in gfa_ids}
        log_input['n_detected_per_gfa'] = {i: 0 for i in gfa_ids}

        att_data.to_hdf(att_dir / f'data{prefix}.in.h5', 'data')

        gfa_xyls = []

        for gfa_id in att_data.gfa.unique():

            gfa_table = astropy.table.Table.from_pandas(
                att_data.loc[(att_data.gfa == gfa_id) & att_data.detected])

            n_stars_gfa = len(att_data.loc[(att_data.gfa == gfa_id)])
            n_detected = len(gfa_table)

            gfa_table.write(att_dir / f'gfa{gfa_id}{prefix}.xyls',
                            format='fits', overwrite=True)
            gfa_xyls.append(str(att_dir / f'gfa{gfa_id}{prefix}.xyls'))

            gfa_id = int(gfa_id)  # To avoid YAML serialising as numpy object
            log_input['n_stars_per_gfa'][gfa_id] = n_stars_gfa
            log_input['n_detected_per_gfa'][gfa_id] = n_detected

        shutil.copy(astrometry_cfg, att_dir)

        with open(att_dir / f'config{prefix}.yaml', 'w') as out:
            out.write(yaml.dump(log_config))

        plate_scale = config_data[observatory]['plate_scale']
        pixel_size = config_data['gfa']['pixel_size']
        pixel_scale = pixel_size / 1000. / plate_scale * 3600.  # In arcsec

        astrometry_net = AstrometryNet()
        astrometry_net.configure(
            backend_config=att_dir / astrometry_cfg.name,
            width=config_data['gfa']['shape'][0],
            height=config_data['gfa']['shape'][1],
            sort_column=config_data['mag_column'],
            sort_ascending=True,
            no_plots=True,
            ra=boresight[0] + (numpy.random.uniform() - 0.5),
            dec=boresight[1] + (numpy.random.uniform() - 0.5),
            radius=2,
            scale_low=pixel_scale * 0.9,
            scale_high=pixel_scale * 1.1,
            scale_units='arcsecperpix',
            dir=att_dir)

        astrometry_net.run(gfa_xyls,
                           stdout=att_dir / f'stdout{prefix}',
                           stderr=att_dir / f'stderr{prefix}')

        log_config['output'] = {}
        log_output = log_config['output']

        att_data['ra_solved'] = numpy.nan
        att_data['dec_solved'] = numpy.nan
        att_data['separation'] = numpy.nan

        log_output['solved'] = {i: False for i in gfa_ids}
        for gfa_id in gfa_ids:
            if not (att_dir / f'gfa{gfa_id}{prefix}.solved').exists():
                continue
            log_output['solved'][gfa_id] = True
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                solved_wcs = astropy.wcs.WCS(str(att_dir /
                                                 f'gfa{gfa_id}{prefix}.wcs'))

            gfa_idx = att_data.gfa == gfa_id
            radec_solved = solved_wcs.wcs_pix2world(
                att_data.loc[gfa_idx, ['x', 'y']].to_numpy(), 0)
            att_data.loc[gfa_idx, 'ra_solved'] = radec_solved[:, 0]
            att_data.loc[gfa_idx, 'dec_solved'] = radec_solved[:, 1]
            att_data.loc[gfa_idx, 'separation'] = sky_separation(
                att_data.loc[gfa_idx, 'ra'], att_data.loc[gfa_idx, 'dec'],
                att_data.loc[gfa_idx, 'ra_solved'],
                att_data.loc[gfa_idx, 'dec_solved'],
            )

        with open(att_dir / f'config{prefix}.yaml', 'w') as out:
            out.write(yaml.dump(log_config))

        att_data.to_hdf(att_dir / f'data{prefix}.out.h5', 'data')


def simulate(n_fields, output_dir, observatory='apo', config_file=None,
             n_cpus=None, n_attempts=10, database_profile=None):
    """Runs a simulation using multiprocessing.

    Note that this function will not work from an interactive interpreter
    since it uses multiprocessing.

    Parameters
    ----------
    n_fields : int
        Number of uniformly distributed fields to test.
    output_dir : str
        The root of the directory structure where all the output files will
        be stored.
    observatory : str
        The observatory, either ``'apo'`` or ``'lco'``.
    config_file : str
        The path to the configuration file for the simulation.
    n_cpus : int
        Number of CPUs to use. If not defined, uses all the CPUs.
    n_attempts : int
        Number of attempts, with randomised noise, to try per field.
    database_profile : str
        The database profile to use, if any.

    """

    n_cpus = n_cpus or multiprocessing.cpu_count()

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config_file:
        config_data = read_yaml_file(config_file)
    else:
        config_data = config.copy()

    numpy.random.seed(config_data['seed'])

    fields = get_uniform_ra_dec(n_fields)

    f = functools.partial(_do_one_field, fields, config_data,
                          observatory, output_dir, n_attempts,
                          database_profile=database_profile)

    with multiprocessing.Pool(processes=n_cpus) as pool:
        pool.map(f, range(len(fields)))
