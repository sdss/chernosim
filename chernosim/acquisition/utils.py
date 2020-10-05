#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2020-09-14
# @Filename: utils.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import astropy.wcs
import numpy
import pandas
import psycopg2

from .. import config


def query_field(boresight, r1=None, r2=None, observatory='apo',
                mag_range=None, mag_column=None, database_params=None):
    """Selects Gaia DR2 stars for a field, from the database.

    Parameters
    ----------
    boresight : tuple
        A tuple with the right ascension and declination of the boresight,
        in degrees.
    r1,r2 : float
        The internal and external radii along which the GFAs are located, in
        degrees.
    observatory : str
        The observatory, used to load the default configuration for the GFAs.
    mag_range : tuple
        The range of magnitudes used to select stars.
    mag_column : str
        The name of the magnitude column to query.
    database_params : dict
        A dictionary of database parameters to create the connection. Can
        include ``user``, ``host``, ``port``, and ``dbname``.

    Returns
    -------
    `~pandas.Dataframe`
        A dataframe with the selected stars.

    """

    obs_data = config[observatory]
    r1 = r1 or obs_data['r1']
    r2 = r2 or obs_data['r2']
    mag_range = mag_range or config['mag_range']
    mag_column = mag_column or config['mag_column']

    query = ('WITH x AS MATERIALIZED (SELECT source_id, ra, dec, '
             '{mag_column}, pmra, pmdec '
             'FROM gaia_dr2_source WHERE '
             'q3c_radial_query(ra, dec, {ra}, {dec}, {r2}) AND '
             'NOT q3c_radial_query(ra, dec, {ra}, {dec}, {r1})) '
             'SELECT * FROM x WHERE {mag_column} > {g_min} AND '
             '{mag_column} < {g_max};')

    query = query.format(ra=boresight[0], dec=boresight[1], r1=r1, r2=r2,
                         g_min=mag_range[0], g_max=mag_range[1],
                         mag_column=mag_column)

    if database_params is None:
        database_params = config['database']

    conn_str = ''
    for key in database_params:
        conn_str += f'{key}={database_params[key]} '

    connection = psycopg2.connect(conn_str)
    data = pandas.read_sql(query, connection)
    connection.close()

    return data


def get_gfa_centre(rot, boresight, observatory='apo', r1=None, r2=None):
    """Determines the coordinates of the centre of a GFA.

    Parameters
    ----------
    rot : float
        The rotation of the GFA, with respect to the boresight, in degrees.
    boresight : tuple
        A tuple with the right ascension and declination of the boresight,
        in degrees.
    observatory : str
        The observatory, used to load the default configuration for the GFAs.
    r1,r2 : float
        The internal and external radii along which the GFAs are located, in
        degrees.

    Returns
    -------
    `tuple`
        A tuple with the ``(RA, Dec)`` coordinates of the centre of the GFA.

    """

    # We'll work on radians until w return.

    obs_data = config[observatory]
    r1 = r1 or obs_data['r1']
    r2 = r2 or obs_data['r2']

    rot = numpy.radians(rot)
    r = numpy.radians(0.5 * (r1 + r2))

    b_ra = numpy.radians(boresight[0])
    b_dec = numpy.radians(boresight[1])

    # Determine the declination of the centre.
    c_dec = numpy.arcsin(
        numpy.sin(b_dec) * numpy.cos(r) +
        numpy.cos(b_dec) * numpy.sin(r) * numpy.cos(rot)
    )

    # Determine the delta RA with respect to the boresight.
    delta_ra = numpy.arcsin(
        numpy.sin(r) * numpy.sin(rot) / numpy.cos(c_dec)
    )

    c_ra = b_ra + delta_ra

    return numpy.degrees([c_ra, c_dec])


def create_gfa_wcs(rot, boresight, observatory='apo',
                   r1=None, r2=None, shape=None, pixel_size=None,
                   plate_scale=None):
    """Creates a mock WCS transformation for a given GFA.

    Parameters
    ----------
    rot : float
        The rotation of the GFA, with respect to the boresight, in degrees.
    boresight : tuple
        A tuple with the right ascension and declination of the boresight,
        in degrees.
    observatory : str
        The observatory, used to load the default configuration for the GFAs.
    r1,r2 : float
        The internal and external radii along which the GFAs are located, in
        degrees.
    shape : tuple
        Number of pixels, in the x and y direction of the GFA chip.
    pixel_size : float
        The pixel size, in microns.
    plate_scale : float
        The plate scale, in mm/deg.

    Returns
    -------
    `~astropy.wcs.WCS`
        The `~astropy.wcs.WCS` object with the WCS representation.

    """

    obs_data = config[observatory]
    r1 = r1 or obs_data['r1']
    r2 = r2 or obs_data['r2']
    plate_scale = plate_scale or obs_data['plate_scale']
    shape = shape or config['gfa']['shape']
    pixel_size = pixel_size or config['gfa']['pixel_size']

    gfa_centre = get_gfa_centre(rot, boresight, observatory=observatory,
                                r1=r1, r2=r2)
    rot_rad = numpy.radians(rot)
    pixel_scale = pixel_size / 1000. / plate_scale

    # Create a WCS rotated rot degrees
    wcs = astropy.wcs.WCS(naxis=2)
    wcs.wcs.crpix = numpy.array(shape) / 2.
    wcs.wcs.cd = numpy.array([[numpy.cos(rot_rad), -numpy.sin(rot_rad)],
                              [numpy.sin(rot_rad), numpy.cos(rot_rad)]]).T
    wcs.wcs.cd *= pixel_scale
    wcs.wcs.crval = gfa_centre
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs.wcs.cunit = ['deg', 'deg']

    return wcs


def get_uniform_ra_dec(n):
    """Selects n points uniformly distributed on a sphere.

    Parameters
    ----------
    n : int
        Number of uniformly distributed points to return.

    Returns
    -------
    `~numpy.ndarray`
        Array of longitude and latitude (or RA/Dec) values on the sphere.

    """

    picked = numpy.zeros((n, 2), dtype=numpy.float64)

    for ii in range(n):
        v = numpy.array([0., 0., 0.])

        x = numpy.random.normal()
        y = numpy.random.normal()
        z = numpy.random.normal()

        v = (x, y, z)
        v = v / numpy.linalg.norm(v)

        ra = numpy.degrees(numpy.arctan2(y, x)) % 360.
        dec = 90 - numpy.degrees(numpy.arctan2(numpy.sqrt(x**2 + y**2), z))

        picked[ii] = (ra, dec)

    return picked


def sky_separation(ra1, dec1, ra2, dec2):
    """Returns the separation between two coordinate sets, in degrees."""

    return numpy.degrees(
        numpy.arccos(
            numpy.sin(numpy.radians(dec1)) * numpy.sin(numpy.radians(dec2)) +
            numpy.cos(numpy.radians(dec1)) * numpy.cos(numpy.radians(dec2)) *
            numpy.cos(numpy.radians(ra1 - ra2))
        )
    )


def get_wcs_rotation(wcs):
    """Returns the rotation of a WCS frame.

    Parameters
    ----------
    wcs : ~astropy.wcs.WCS
        The `~astropy.wcs.WCS` object.

    Return
    ------
    float
        The rotation angle, in degrees.

    """

    return numpy.degrees(numpy.arctan2(wcs.wcs.cd[1, 0], wcs.wcs.cd[0, 0]))
