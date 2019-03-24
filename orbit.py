"""
A package including tools for numerical integration of orbits.
"""

from __future__ import (
    division, print_function, absolute_import, unicode_literals)

import numpy as np
from numpy import ndarray
from astropy.utils.console import ProgressBar


# --------------------------------------------------------------------


def simulateOrbit(x0, xdot0, calcAccel, dt, tmax, method='leapfrog',
                  fname="", dtout=0.02, ancout=(), anckw={},
                  verbose=False, **kw):

    """
    Simulate the orbit of particle(s) in given potential well.

    Parameters
    ----------
    x0 : ndarray
        Initial position of the particle(s)
        shape : (n_dimension, ) or (n_particle, n_dimension)
    xdot0 : ndarray
        Initial velocity of the particle(s)
        shape : (n_dimension, ) or (n_particle, n_dimension)
    calcAccel : function
        Function defining the acceleration 'xddot' for particle(s)
        with given position, velocity and time coordinates.
    dt : scalar
        Step size in time
    tmax : scalar
        End point in time
    method : {'leapfrog', 'Euler'}, optional
        Type of the numerical integration method to use
        (default='leapfrog')
    fname : string, optional
        Name of the output file
    dtout : scalar, optional
        Step size in time for outputs
    ancout : tuple, optional
        Ancillary quantities to be outputed into the record array
        Should be tuple of functions, each taking position, velocity
        and time as required arguments.
    anckw : dict, optional
        Keyword arguments to be passed through to each function in
        `ancout`
    verbose : bool, optional
        Whether to show the progress bar (default=False)
    **kw : 
        Other keyword arguments to be passed through to `calcAccel`

    Returns
    -------
    record : ndarray
    Array containing the recorded time, position, velocity,
    and (optionally) other user specified ancillary quantities.
    Information is recorded once per `dtout`, starting from t=0.
    shape : (n_stepout, 1+n_dimension*n_particle*2+?)
    """

    if x0.ndim == 1 and xdot0.ndim == 1:
        n_part = 1
        n_dim = x0.size
    elif x0.ndim == 2 and xdot0.ndim == 2:
        n_part, n_dim = x0.shape
    else:
        raise ValueError("Illegal shape for `x0` or `xdot0`!")

    t = 0.0
    x = x0.copy()
    xdot = xdot0.copy()
    xddot = calcAccel(x, xdot, t, **kw)

    ancs = np.zeros(len(ancout))
    for i, func in enumerate(ancout):
        ancs[i] = func(x, xdot, t, **anckw)

    record = np.empty([int(tmax/dtout)+1,
                       1+len(x)+len(xdot)+len(ancout)])
    tout = dtout
    irec = 0
    record[irec, :] = np.hstack([t, ndarray.flatten(x),
                                 ndarray.flatten(xdot), ancs])

    if verbose:
        bar = ProgressBar(int(tmax/dt))
    while t < tmax:
        t += dt
        if method == 'Euler':
            x += xdot * dt
            xdot += xddot * dt
            xddot = calcAccel(x, xdot, t, **kw)
        elif method == 'leapfrog':  # also called 'kick-drift-kick'
            xdot += xddot * dt / 2
            x += xdot * dt
            xddot = calcAccel(x, xdot, t, **kw)
            xdot += xddot * dt / 2
        else:
            raise ValueError("`method` must be either 'Euler'"
                             " or 'leapfrog'.")
        if t >= tout:
            tout += dtout
            irec += 1
            for i, func in enumerate(ancout):
                ancs[i] = func(x, xdot, t, **anckw)
            record[irec, :] = np.hstack([t, ndarray.flatten(x),
                                         ndarray.flatten(xdot),
                                         ancs])
        if verbose:
            bar.update()
    print("")

    if fname:
        fmt = (['%.8g'] + ['%.8g'] * x0.size +
               ['%.8g'] * x0.size + ['%.8g'])
        np.savetxt(fname, record, fmt, delimiter=' ')

    return record


# --------------------------------------------------------------------


def calcKE(x, xdot, t, m=1.0, **kw):
    """
    Calculate the total kinetic energy for particle(s).

    Parameters
    ----------
    x : ndarray
        Position of the particles(s)
        (Note: This variable does not enter the calculation.
         It is included simply for compatibility concerns.)
        shape : (n_dimension, ) or (n_particle, n_dimension)
    xdot : ndarray
        Velocity of the particle(s)
        shape : (n_dimension, ) or (n_particle, n_dimension)
    t : scalar
        Time
        (Note: This variable does not enter the calculation.
         It is included simply for compatibility concerns.)
    m : scalar or ndarray
        Mass of the particle(s) (default=1.0)
        shape : scalar or (n_particle, )
    
    Returns
    -------
    K : scalar
        Total kinetic energy.
    """
    return 0.5 * np.sum(m * np.sum(xdot**2, -1))


# --------------------------------------------------------------------


def calcAccel_Kepler(x, xdot, t, m=1.0, e=1.0, k=1.0, **kw):
    """
    Calculate the acceleration for particles in a Keplerian potential.

    Parameters
    ----------
    x : ndarray
        Position of the particle(s)
        shape : (n_dimension, ) or (n_particle, n_dimension)
    xdot : ndarray
        Velocity of the particle(s)
        (Note: This variable does not enter the calculation.
         It is included simply for compatibility concerns.)
        shape : (n_dimension, ) or (n_particle, n_dimension)
    t : scalar
        Time
        (Note: This variable does not enter the calculation.
         It is included simply for compatibility concerns.)
    m : scalar or ndarray
        Inertial mass of the particle(s) (default=1.0)
        shape : scalar or (n_particle, )
    e : scalar or ndarray
        Coupling constant of the particle(s) with the potential
        (default=1.0)
        shape : scalar or (n_particle, )
    k : scalar, optional
        Normalization constant (default=1.0)

    Returns
    -------
    xddot : ndarray
        Acceleration array of the particle(s), which has the same
        shape as `x`.

    See Also
    --------
    calcUE_Kepler
    
    Notes
    -----
    Acceleration due to Keplerian potential:
    .. math:: a_i = - \frac{k e x_i}{m r^3} ,
    where :math:`a_i` is the acceleration of a particle along the
    i-th dimension, :math:`x_i` its position on the i-th dimension,
    and r the distance from the origin.
    """
    return -(k * e / m * x.T / np.sqrt(np.sum(x**2, -1))**3).T


# --------------------------------------------------------------------


def calcUE_Kepler(x, xdot, t, m=1.0, e=1.0, k=1.0, **kw):
    """
    Calculate the total potential energy for particles in a Keplerian
    potential.

    x : ndarray
        Position of the particle(s)
        shape : (n_dimension, ) or (n_particle, n_dimension)
    xdot : ndarray
        Velocity of the particle(s)
        (Note: This variable does not enter the calculation.
         It is included simply for compatibility concerns.)
        shape : (n_dimension, ) or (n_particle, n_dimension)
    t : scalar
        Time
        (Note: This variable does not enter the calculation.
         It is included simply for compatibility concerns.)
    m : scalar or ndarray
        Inertial mass of the particle(s) (default=1.0)
        shape : scalar or (n_particle, )
    e : scalar or ndarray
        Coupling constant of the particle(s) with the potential
        (default=1.0)
        shape : scalar or (n_particle, )
    k : scalar, optional
        Normalization constant (default=1.0)

    Returns
    -------
    U : scalar
        Total potential energy of the particle(s)

    See Also
    --------
    calcAccel_Kepler
    
    Notes
    -----
    Keplerian potential energy:
    .. math:: U = - k e / r ,
    where r is the distance from the origin.
    """
    return -np.sum(k * e / np.sqrt(np.sum(x**2, -1)))


# --------------------------------------------------------------------


def calcAccel_HO(x, xdot, t, m=1.0, e=1.0, q=1.0, **kw):
    """
    Calculate the acceleration for particles in a Harmonic Oscillator
    potential.

    Parameters
    ----------
    x : ndarray
        Position of the particle(s)
        shape : (n_dimension, ) or (n_particle, n_dimension)
    xdot : ndarray
        Velocity of the particle(s)
        (Note: This variable does not enter the calculation.
         It is included simply for compatibility concerns.)
        shape : (n_dimension, ) or (n_particle, n_dimension)
    t : scalar
        Time
        (Note: This variable does not enter the calculation.
         It is included simply for compatibility concerns.)
    m : scalar or ndarray
        Inertial mass of the particle(s) (default=1.0)
        shape : scalar or (n_particle, )
    e : scalar or ndarray
        Coupling constant of the particle(s) with the potential
        (default=1.0)
        shape : scalar or (n_particle, )
    q : ndarray, optional
        Normalization constant for each dimension (default=1.0)
        shape : (n_dimension, )

    Returns
    -------
    xddot : ndarray
        Acceleration array of the particle(s), which has the same
        shape as `x`.

    See Also
    --------
    calcUE_HO
    
    Notes
    -----
    Acceleration due to Harmonic Oscillator potential:
    .. math:: a_i = - \frac{e x_i}{m q_i^2} ,
    where :math:`a_i` is the acceleration of a particle along the
    i-th dimension, :math:`x_i` its position on the i-th dimension,
    and :math:`q_i` the corresponding scale length.
    """
    return - (e / m * (x/q**2).T).T


# --------------------------------------------------------------------


def calcUE_HO(x, xdot, t, m=1.0, e=1.0, q=1.0, **kw):
    """
    Calculate the total potential energy for particles in a
    Harmonic Oscillator potential.

    Parameters
    ----------
    x : ndarray
        Position of the particle(s)
        shape : (n_dimension, ) or (n_particle, n_dimension)
    xdot : ndarray
        Velocity of the particle(s)
        (Note: This variable does not enter the calculation.
         It is included simply for compatibility concerns.)
        shape : (n_dimension, ) or (n_particle, n_dimension)
    t : scalar
        Time
        (Note: This variable does not enter the calculation.
         It is included simply for compatibility concerns.)
    m : scalar or ndarray
        Inertial mass of the particle(s) (default=1.0)
        shape : scalar or (n_particle, )
    e : scalar or ndarray
        Coupling constant of the particle(s) with the potential
        (default=1.0)
        shape : scalar or (n_particle, )
    q : ndarray, optional
        Normalization constant for each dimension (default=1.0)
        shape : (n_dimension, )

    Returns
    -------
    U : scalar
        Total potential energy of the particle(s)

    See Also
    --------
    calcAccel_HO
    
    Notes
    -----
    Harmonic Oscillator potential:
    .. math:: U = \Sigma_i \frac{e x_i^2}{2 q_i^2} ,
    where :math:`x_i` its position on the i-th dimension,
    and :math:`q_i` the corresponding scale length.
    """
    return 0.5 * np.sum(e * np.sum((x/q)**2, -1))


# --------------------------------------------------------------------


def calcAccel_NBodyGravity(x, xdot, t, m=1.0, G=1.0, **kw):
    """
    Calculate the acceleration for particles due to their mutual
    gravitational interaction.

    Parameters
    ----------
    x : ndarray
        Position of the particles
        shape : (n_particle, n_dimension)
    xdot : ndarray
        Velocity of the particles
        (Note: This variable does not enter the calculation.
         It is included simply for compatibility concerns.)
        shape : (n_particle, n_dimension)
    t : scalar
        Time
        (Note: This variable does not enter the calculation.
         It is included simply for compatibility concerns.)
    m : scalar or ndarray, optional
        Mass of the particles (default=1.0)
        shape : scalar or (n_particle, )
    G : scalar, optional
        Gravitational constant (default=1.0)

    Returns
    -------
    xddot : ndarray
        Acceleration array of the particles, which has the same
        shape as `x`.

    See Also
    --------
    calcUE_NBodyGravity

    Notes
    -----
    Gravitational acceleration:
    .. math:: a_i = \Sigma_{j \neq i} \frac{G m_j (x_j - x_i)}{r_{ij}^3} ,
    where :math:`a_i` is the acceleration of a particle along the i-th
    dimension, :math:`x_i` its position on the i-th dimension, and
    :math:`r_{ij}` the distance between the i-th and j-th particle.
    """
    n_part, n_dim = x.shape
    a = np.zeros((n_part, n_dim))
    dxij = x.reshape(n_part, 1, n_dim) - x.reshape(1, n_part, n_dim)
    rij = np.sqrt((dxij**2).sum(2))
    rij[rij == 0] = np.inf
    a = G * (m.reshape(n_part, 1, 1) *
             dxij / rij.reshape(n_part, n_part, 1)**3).sum(0)
    return a


# --------------------------------------------------------------------


def calcUE_NBodyGravity(x, xdot, t, m=1.0, G=1.0, **kw):
    """
    Calculate the total potential energy for particles with mutual
    gravitational interaction.

    Parameters
    ----------
    x : ndarray
        Position of the particles
        shape : (n_particle, n_dimension)
    xdot : ndarray
        Velocity of the particles
        (Note: This variable does not enter the calculation.
         It is included simply for compatibility concerns.)
        shape : (n_particle, n_dimension)
    t : scalar
        Time
        (Note: This variable does not enter the calculation.
         It is included simply for compatibility concerns.)
    m : scalar or ndarray, optional
        Mass of the particles (default=1.0)
        shape : scalar or (n_particle, )
    G : scalar, optional
        Gravitational constant (default=1.0)

    Returns
    -------
    U : scalar
        Total potential energy of the particles

    See Also
    --------
    calcAccel_NBodyGravity

    Notes
    -----
    Gravitational potential energy:
    .. math:: U = - \Sigma_{j \neq i} \frac{G m_i m_j}{r_{ij}} ,
    where :math:`m_i` is the mass of the i-th particle, and
    :math:`r_{ij}` the distance between the i-th and j-th particle.
    """
    n_part, n_dim = x.shape
    dxij = x.reshape(n_part, 1, n_dim) - x.reshape(1, n_part, n_dim)
    rij = np.sqrt((dxij**2).sum(2))
    rij[rij == 0] = np.inf
    U = -G * np.sum(m.reshape(n_part, 1) *
                    m.reshape(1, n_part) / rij) / 2
    return U


# --------------------------------------------------------------------


# if __name__ == '__main__':

    # import itertools

    # Kepler Potential
    # print("Kepler Potential")
    # x0 = np.array([1., 0.])
    # xdot0 = np.array([0., 1.])
    # tmax = 40.
    # dt = 1e-4
    # for vy, dt in itertools.product((1., .5, .2), (1e-2, 1e-4)):
    #     print("ydot0={}, dt={}".format(vy, dt))
    #     xdot0[1] = vy
    #     fname = "data/kep.ydot{}.eul.dt{}".format(xdot0[1], dt)
    #     simulateOrbit(x0, xdot0, calcAccel_Kepler, dt, tmax,
    #                   fname=fname, method='Euler')
    #     fname = "data/kep.ydot{}.leap.dt{}".format(xdot0[1], dt)
    #     simulateOrbit(x0, xdot0, calcAccel_Kepler, dt, tmax,
    #                   fname=fname)

    # # Harmonic Oscillator Potential
    # print("Harmonic Oscillator Potential")
    # x0 = np.array([1., 0.])
    # xdot0 = np.array([0., 1.])
    # q = np.array([1., 1.])
    # tmax = 60.
    # dt = 1e-4
    # for vy, qy in itertools.product((1., .5), (1., .9, .6)):
    #     print("ydot0={}, q={}".format(vy, qy))
    #     xdot0[1] = vy
    #     q[1] = qy
    #     fname = "data/ho.q{}.ydot{}.leap.dt{}".format(q[1], xdot0[1], dt)
    #     simulateOrbit(x0, xdot0, calcAccel_HO, dt, tmax, q=q,
    #                   fname=fname)
    # xdot0 = np.array([0., .5])
    # q = np.array([1., .9])
    # tmax = 40.
    # dt = 1e-4
    # for dt in (1e-2, 1e-3, 1e-4):
    #     print("dt={}".format(dt))
    #     fname = "data/ho.q{}.ydot{}.eul.dt{}".format(q[1], xdot0[1], dt)
    #     simulateOrbit(x0, xdot0, calcAccel_HO, dt, tmax, q=q,
    #                   fname=fname, method='Euler')
    #     fname = "data/ho.q{}.ydot{}.leap.dt{}".format(q[1], xdot0[1], dt)
    #     simulateOrbit(x0, xdot0, calcAccel_HO, dt, tmax, q=q,
    #                   fname=fname)

    # # Planet and Moon
    # print("Planet and Moon")
    # tmax = 15.
    # dt = 1e-4
    # G = 1.
    # m = np.array([1., .1, .01])
    # x0 = np.array([[0., 0.], [1., 0.], [1.05, 0.]])
    # relvp = 1.0
    # relvm = 1.0
    # for xm, tmax in zip((1.05, 1.1, 1.2), (15., 15., 10.)):
    #     x0[2, 0] = xm
    #     info = "mp{}.mm{}.xm{:.2f}.ydp{:.1f}.ydm{:.1f}"
    #     info = info.format(m[1], m[2], x0[2, 0], relvp, relvm)
    #     print(info)
    #     fname = "data/moon." + info
    #     # Distance from the mass center of the planet-moon system
    #     # to the star
    #     rsp = np.sqrt(np.sum(((x0.T * m)[:, 1:].sum(1) /
    #                           m[1:].sum() - x0[0, :])**2))
    #     # Relative velocity between the planet-moon system and the star
    #     vsp = np.sqrt(G * m.sum() / rsp)
    #     # Distance between the moon and the planet
    #     rpm = np.sqrt(np.sum((x0[1, :] - x0[2, :])**2))
    #     # Relative velocity between the moon and the planet
    #     vpm = np.sqrt(G * m[1:].sum() / rpm)
    #     # Velocity array in the frame of the mass center
    #     xdot0 = np.zeros_like(x0)
    #     xdot0[2, 1] = vpm * relvm
    #     xdot0[1:, :] -= (xdot0.T * m)[:, 1:].sum(1) / m[1:].sum()
    #     xdot0[1:, 1] += vsp * relvp
    #     xdot0 -= (xdot0.T * m).sum(1) / m.sum()
    #     # # Position of the center of mass
    #     # xc = (x0.T * m).sum(1) / m.sum()
    #     # # Position array centered at the center of mass
    #     # x0 -= xc
    #     simulateOrbit(x0, xdot0, calcAccel_NBodyGravity, dt, tmax,
    #                   m=m, e=m, G=G, fname=fname)

    # # Slow Planet and Moon
    # print("Slow Planet and Moon")
    # tmax = 8.
    # dt = 1e-5
    # dtout = 0.001
    # G = 1.
    # m = np.array([1., .1, .01])
    # x0 = np.array([[0., 0.], [1., 0.], [1.05, 0.]])
    # relvp = .7
    # relvm = 1.
    # info = "mp{}.mm{}.xm{:.2f}.ydp{:.1f}.ydm{:.1f}"
    # info = info.format(m[1], m[2], x0[2, 0], relvp, relvm)
    # print(info)
    # fname = "data/moon." + info
    # # Distance from the mass center of the planet-moon system
    # # to the star
    # rsp = np.sqrt(np.sum(((x0.T * m)[:, 1:].sum(1) /
    #                       m[1:].sum())**2))
    # # Relative velocity between the planet-moon system and the star
    # vsp = np.sqrt(G * m.sum() / rsp)
    # # Distance between the moon and the planet
    # rpm = np.sqrt(np.sum((x0[1, :] - x0[2, :])**2))
    # # Relative velocity between the moon and the planet
    # vpm = np.sqrt(G * m[1:].sum() / rpm)
    # # Velocity array in the frame of the mass center
    # xdot0 = np.zeros_like(x0)
    # xdot0[2, 1] = vpm * relvm
    # xdot0[1:, :] -= (xdot0.T * m)[:, 1:].sum(1) / m[1:].sum()
    # xdot0[1:, 1] += vsp * relvp
    # xdot0 -= (xdot0.T * m).sum(1) / m.sum()
    # # # Position of the center of mass
    # # xc = (x0.T * m).sum(1) / m.sum()
    # # # Position array centered at the center of mass
    # # x0 -= xc
    # simulateOrbit(x0, xdot0, calcAccel_NBodyGravity, dt, tmax,
    #               m=m, e=m, G=G, fname=fname, dtout=dtout)
    # tmax = 15.
    # dt = 2e-5
    # x0 = np.array([[0., 0.], [1., 0.], [1.05, 0.]])
    # relvp = 1.
    # relvm = .7
    # info = "mp{}.mm{}.xm{:.2f}.ydp{:.1f}.ydm{:.1f}"
    # info = info.format(m[1], m[2], x0[2, 0], relvp, relvm)
    # print(info)
    # fname = "data/moon." + info
    # # Distance from the mass center of the planet-moon system
    # # to the star
    # rsp = np.sqrt(np.sum(((x0.T * m)[:, 1:].sum(1) /
    #                       m[1:].sum())**2))
    # # Relative velocity between the planet-moon system and the star
    # vsp = np.sqrt(G * m.sum() / rsp)
    # # Distance between the moon and the planet
    # rpm = np.sqrt(np.sum((x0[1, :] - x0[2, :])**2))
    # # Relative velocity between the moon and the planet
    # vpm = np.sqrt(G * m[1:].sum() / rpm)
    # # Velocity array in the frame of the mass center
    # xdot0 = np.zeros_like(x0)
    # xdot0[2, 1] = vpm * relvm
    # xdot0[1:, :] -= (xdot0.T * m)[:, 1:].sum(1) / m[1:].sum()
    # xdot0[1:, 1] += vsp * relvp
    # xdot0 -= (xdot0.T * m).sum(1) / m.sum()
    # # # Position of the center of mass
    # # xc = (x0.T * m).sum(1) / m.sum()
    # # # Position array centered at the center of mass
    # # x0 -= xc
    # simulateOrbit(x0, xdot0, calcAccel_NBodyGravity, dt, tmax,
    #               m=m, e=m, G=G, fname=fname, dtout=dtout)
    # x0 = np.array([[0., 0.], [1., 0.], [1.1, 0.]])
    # relvp = 1.
    # relvm = -1.
    # info = "mp{}.mm{}.xm{:.2f}.ydp{:.1f}.ydm{:.1f}"
    # info = info.format(m[1], m[2], x0[2, 0], relvp, relvm)
    # print(info)
    # fname = "data/moon." + info
    # # Distance from the mass center of the planet-moon system
    # # to the star
    # rsp = np.sqrt(np.sum(((x0.T * m)[:, 1:].sum(1) /
    #                       m[1:].sum())**2))
    # # Relative velocity between the planet-moon system and the star
    # vsp = np.sqrt(G * m.sum() / rsp)
    # # Distance between the moon and the planet
    # rpm = np.sqrt(np.sum((x0[1, :] - x0[2, :])**2))
    # # Relative velocity between the moon and the planet
    # vpm = np.sqrt(G * m[1:].sum() / rpm)
    # # Velocity array in the frame of the mass center
    # xdot0 = np.zeros_like(x0)
    # xdot0[2, 1] = vpm * relvm
    # xdot0[1:, :] -= (xdot0.T * m)[:, 1:].sum(1) / m[1:].sum()
    # xdot0[1:, 1] += vsp * relvp
    # xdot0 -= (xdot0.T * m).sum(1) / m.sum()
    # # # Position of the center of mass
    # # xc = (x0.T * m).sum(1) / m.sum()
    # # # Position array centered at the center of mass
    # # x0 -= xc
    # simulateOrbit(x0, xdot0, calcAccel_NBodyGravity, dt, tmax,
    #               m=m, e=m, G=G, fname=fname, dtout=dtout)
    # x0 = np.array([[0., 0.], [1., 0.], [1.1, 0.]])
    # relvm = -.7
    # info = "mp{}.mm{}.xm{:.2f}.ydp{:.1f}.ydm{:.1f}"
    # info = info.format(m[1], m[2], x0[2, 0], relvp, relvm)
    # print(info)
    # fname = "data/moon." + info
    # # Distance from the mass center of the planet-moon system
    # # to the star
    # rsp = np.sqrt(np.sum(((x0.T * m)[:, 1:].sum(1) /
    #                       m[1:].sum())**2))
    # # Relative velocity between the planet-moon system and the star
    # vsp = np.sqrt(G * m.sum() / rsp)
    # # Distance between the moon and the planet
    # rpm = np.sqrt(np.sum((x0[1, :] - x0[2, :])**2))
    # # Relative velocity between the moon and the planet
    # vpm = np.sqrt(G * m[1:].sum() / rpm)
    # # Velocity array in the frame of the mass center
    # xdot0 = np.zeros_like(x0)
    # xdot0[2, 1] = vpm * relvm
    # xdot0[1:, :] -= (xdot0.T * m)[:, 1:].sum(1) / m[1:].sum()
    # xdot0[1:, 1] += vsp * relvp
    # xdot0 -= (xdot0.T * m).sum(1) / m.sum()
    # # # Position of the center of mass
    # # xc = (x0.T * m).sum(1) / m.sum()
    # # # Position array centered at the center of mass
    # # x0 -= xc
    # simulateOrbit(x0, xdot0, calcAccel_NBodyGravity, dt, tmax,
    #               m=m, e=m, G=G, fname=fname, dtout=dtout)

    # # Realistic Planet and Moon
    # print("Realistic Planet and Moon")
    # tmax = 15.
    # dt = 1e-4
    # G = 1.
    # m = np.array([1., .01, .0001])
    # x0 = np.array([[0., 0.], [1., 0.], [1.05, 0.]])
    # relvp = 1.
    # relvm = 1.
    # for xm in [1.05, 1.08, 1.1]:
    #     x0[2, 0] = xm
    #     info = "mp{}.mm{}.xm{:.2f}.ydp{:.1f}.ydm{:.1f}"
    #     info = info.format(m[1], m[2], x0[2, 0], relvp, relvm)
    #     print(info)
    #     fname = "data/moon." + info
    #     # Distance from the mass center of the planet-moon system
    #     # to the star
    #     rsp = np.sqrt(np.sum(((x0.T * m)[:, 1:].sum(1) /
    #                           m[1:].sum())**2))
    #     # Relative velocity between the planet-moon system and the star
    #     vsp = np.sqrt(G * m.sum() / rsp)
    #     # Distance between the moon and the planet
    #     rpm = np.sqrt(np.sum((x0[1, :] - x0[2, :])**2))
    #     # Relative velocity between the moon and the planet
    #     vpm = np.sqrt(G * m[1:].sum() / rpm)
    #     # Velocity array in the frame of the mass center
    #     xdot0 = np.zeros_like(x0)
    #     xdot0[2, 1] = vpm * relvm
    #     xdot0[1:, :] -= (xdot0.T * m)[:, 1:].sum(1) / m[1:].sum()
    #     xdot0[1:, 1] += vsp * relvp
    #     xdot0 -= (xdot0.T * m).sum(1) / m.sum()
    #     # # Position of the center of mass
    #     # xc = (x0.T * m).sum(1) / m.sum()
    #     # # Position array centered at the center of mass
    #     # x0 -= xc
    #     simulateOrbit(x0, xdot0, calcAccel_NBodyGravity, dt, tmax,
    #                   m=m, e=m, G=G, fname=fname)

    # # The Three-Body Problem!!!
    # print("Hierarchical Triple")
    # tmax = 3 * 2 * np.pi / np.sqrt(3)
    # dt = 1e-5
    # dtout = 0.001
    # G = 1.
    # m = np.ones(3)
    # for rps in [0.1, 0.2, 0.5]:
    #     print("rps={}".format(rps))
    #     fname = "data/threebody.HierarchicalTriple{}".format(rps)
    #     x0 = np.array([[rps/2, 0.], [-rps/2, 0.], [1., 0.]])
    #     # Position of the center of mass
    #     xc = (x0.T * m).sum(1) / m.sum()
    #     # Position array centered at the center of mass
    #     x0 -= xc
    #     xdot0 = np.zeros_like(x0)
    #     # Distance between the primary and the secondary
    #     rps = np.sqrt(np.sum((x0[0, :] - x0[1, :])**2))
    #     # Relative velocity between the primary and the secondary
    #     vps = np.sqrt(G * m[:2].sum() / rps)
    #     # Distance from the tertiary to the inner binary
    #     rbt = np.sqrt(np.sum(((x0.T * m)[:, :2].sum(1) / m[:2].sum() -
    #                           x0[2, :])**2))
    #     # Relative velocity between the tertiary and the inner binary
    #     vbt = np.sqrt(G * m.sum() / rbt)
    #     # Velocity array in the frame of the center of mass
    #     xdot0[1, 1] = vps
    #     xdot0[:2, :] -= (xdot0.T * m)[:, :2].sum(1) / m[:2].sum()
    #     xdot0[2, 1] += vbt
    #     xdot0 -= (xdot0.T * m).sum(1) / m.sum()
    #     simulateOrbit(x0, xdot0, calcAccel_NBodyGravity, dt, tmax,
    #                   m=m, e=m, G=G, fname=fname, dtout=dtout)
    # print("Figure-8")
    # tmax = 6.32591398 / 3
    # dt = 1e-5
    # dtout = 0.001
    # G = 1.
    # m = np.ones(3)
    # fname = "data/threebody.Figure-8"
    # x0 = np.zeros([3, 2])
    # x0[0, :] = [0.97000436, -0.24308753]
    # x0[1, :] = -x0[0, :]
    # xdot0 = np.zeros([3, 2])
    # xdot0[2, :] = [-0.93240737, -0.86473146]
    # xdot0[0, :] = -xdot0[2, :] / 2
    # xdot0[1, :] = -xdot0[2, :] / 2
    # simulateOrbit(x0, xdot0, calcAccel_NBodyGravity, dt, tmax,
    #               m=m, e=m, G=G, fname=fname, dtout=dtout)
    # print("Broucke & Boggs (1973) Solutions")
    # dt = 1e-5
    # dtout = 0.001
    # G = 1.
    # m = np.ones(3) / 3
    # fname = "data/threebody.BB73-1"
    # tmax = 6.061160 * 2
    # x0 = np.zeros([3, 2])
    # xdot0 = np.zeros([3, 2])
    # x0[1, 0] = 1.003649
    # x0[2, 0] = -1.232358
    # x0 -= (x0.T * m).sum(1) / m.sum()
    # xdot0[1, 1] = 1.263550
    # xdot0[2, 1] = 0.794760
    # xdot0 -= (xdot0.T * m).sum(1) / m.sum()
    # simulateOrbit(x0, xdot0, calcAccel_NBodyGravity, dt, tmax,
    #               m=m, e=m, G=G, fname=fname, dtout=dtout)
    # fname = "data/threebody.BB73-4"
    # tmax = 2.134540 * 2
    # x0 = np.zeros([3, 2])
    # xdot0 = np.zeros([3, 2])
    # x0[1, 0] = 0.888347
    # x0[2, 0] = 0.222470
    # x0 -= (x0.T * m).sum(1) / m.sum()
    # xdot0[1, 1] = 0.143591
    # xdot0[2, 1] = -2.077961
    # xdot0 -= (xdot0.T * m).sum(1) / m.sum()
    # simulateOrbit(x0, xdot0, calcAccel_NBodyGravity, dt, tmax,
    #               m=m, e=m, G=G, fname=fname, dtout=dtout)
    # fname = "data/threebody.BB73-6"
    # tmax = 6.668193 * 2
    # x0 = np.zeros([3, 2])
    # xdot0 = np.zeros([3, 2])
    # x0[1, 0] = 1.831744
    # x0[2, 0] = 1.946311
    # x0 -= (x0.T * m).sum(1) / m.sum()
    # xdot0[1, 1] = 1.336857
    # xdot0[2, 1] = 0.236280
    # xdot0 -= (xdot0.T * m).sum(1) / m.sum()
    # simulateOrbit(x0, xdot0, calcAccel_NBodyGravity, dt, tmax,
    #               m=m, e=m, G=G, fname=fname, dtout=dtout)
    # fname = "data/threebody.BB73-40"
    # tmax = 4 * np.pi
    # x0 = np.zeros([3, 2])
    # xdot0 = np.zeros([3, 2])
    # x0[1, 0] = 0.840381
    # x0[2, 0] = 2.037556
    # x0 -= (x0.T * m).sum(1) / m.sum()
    # xdot0[1, 1] = 0.922163
    # xdot0[2, 1] = -0.356653
    # xdot0 -= (xdot0.T * m).sum(1) / m.sum()
    # simulateOrbit(x0, xdot0, calcAccel_NBodyGravity, dt, tmax,
    #               m=m, e=m, G=G, fname=fname, dtout=dtout)
    # fname = "data/threebody.BB73-122"
    # tmax = 4.374745 * 2
    # x0 = np.zeros([3, 2])
    # xdot0 = np.zeros([3, 2])
    # x0[1, 0] = 0.923743
    # x0[2, 0] = -0.870149
    # x0 -= (x0.T * m).sum(1) / m.sum()
    # xdot0[1, 1] = 0.810866
    # xdot0[2, 1] = -0.164107
    # xdot0 -= (xdot0.T * m).sum(1) / m.sum()
    # simulateOrbit(x0, xdot0, calcAccel_NBodyGravity, dt, tmax,
    #               m=m, e=m, G=G, fname=fname, dtout=dtout)

