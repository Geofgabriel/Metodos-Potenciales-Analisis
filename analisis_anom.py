# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 12:13:32 2021

@author: Gabriel R. Gelpi
"""
from __future__ import division, absolute_import
import warnings
import numpy
import numpy as np #
import utils



#====================================================
def hga(x, y, data, shape, method='fd'):
    
    dx = derivx(x, y, data, shape, method=method)
    dy = derivy(x, y, data, shape, method=method)
    
    res = numpy.sqrt(dx ** 2 + dy ** 2)
    return res


#=====================================================
def butt(x, y, data, shape, lamb_c, n):
    r"""
    Filtro Butterworth
    """
    nx, ny = shape
    # Pad the array with the edge values to avoid instability
    padded, padx, pady = _pad_data(data, shape)
    kx, ky = _fftfreqs(x, y, shape, padded.shape)
    kz = numpy.sqrt(kx**2 + ky**2)
    kc = 2*np.pi/lamb_c
    H = 1/(1+(kz/kc)*n) # filtro
    pb_ft = numpy.fft.fft2(padded)*H # fft*H
    cont = numpy.real(numpy.fft.ifft2(pb_ft))
    # Remove padding
    cont = cont[padx: padx + nx, pady: pady + ny].ravel()
    return cont

#=====================================================
def pb_gaus(x, y, data, shape, lamb_c):
    r"""
    Filtro Gaussiano pasa bajos
    """
    nx, ny = shape
    # Pad the array with the edge values to avoid instability
    padded, padx, pady = _pad_data(data, shape)
    kx, ky = _fftfreqs(x, y, shape, padded.shape)
    
    kz = numpy.sqrt(kx**2 + ky**2)
    
    k12 = 2*np.pi/lamb_c 
    a = k12*k12/np.log(2)
    
    pb_ft = numpy.fft.fft2(padded)*numpy.exp(-kz*kz/a)
    cont = numpy.real(numpy.fft.ifft2(pb_ft))
    # Remove padding
    cont = cont[padx: padx + nx, pady: pady + ny].ravel()
    return cont


#=====================================================
def pa_gaus(x, y, data, shape, lamb_c):
    r"""
    Fiiltro Gaussiano pasa altos
    """
    nx, ny = shape
    # Pad the array with the edge values to avoid instability
    padded, padx, pady = _pad_data(data, shape)
    kx, ky = _fftfreqs(x, y, shape, padded.shape)
    
    kz = numpy.sqrt(kx**2 + ky**2)
    
    k12 = 2*np.pi/lamb_c 
    a = k12*k12/np.log(2)
    pa_ft = numpy.fft.fft2(padded)*(1-numpy.exp(-kz*kz/a))
    cont = numpy.real(numpy.fft.ifft2(pa_ft))
    # Remove padding
    cont = cont[padx: padx + nx, pady: pady + ny].ravel()
    return cont


#=====================================================
def upcontinue(x, y, data, shape, height):
    r"""
    Upward continuation of potential field data.
    Calculates the continuation through the Fast Fourier Transform in the
    wavenumber domain (Blakely, 1996):
    .. math::
        F\{h_{up}\} = F\{h\} e^{-\Delta z |k|}
    and then transformed back to the space domain. :math:`h_{up}` is the upward
    continue data, :math:`\Delta z` is the height increase, :math:`F` denotes
    the Fourier Transform,  and :math:`|k|` is the wavenumber modulus.
    .. note:: Requires gridded data.
    .. note:: x, y, z and height should be in meters.
    .. note::
        It is not possible to get the FFT of a masked grid. The default
        :func:`fatiando.gridder.interp` call using minimum curvature will not
        be suitable.  Use ``extrapolate=True`` or ``algorithm='nearest'`` to
        get an unmasked grid.
    Parameters:
    * x, y : 1D-arrays
        The x and y coordinates of the grid points
    * data : 1D-array
        The potential field at the grid points
    * shape : tuple = (nx, ny)
        The shape of the grid
    * height : float
        The height increase (delta z) in meters.
    Returns:
    * cont : array
        The upward continued data
    References:
    Blakely, R. J. (1996), Potential Theory in Gravity and Magnetic
    Applications, Cambridge University Press.
    """
    
    nx, ny = shape
    # Pad the array with the edge values to avoid instability
    padded, padx, pady = _pad_data(data, shape)
    kx, ky = _fftfreqs(x, y, shape, padded.shape)
    kz = numpy.sqrt(kx**2 + ky**2)
    upcont_ft = numpy.fft.fft2(padded)*numpy.exp(-height*kz)
    cont = numpy.real(numpy.fft.ifft2(upcont_ft))
    # Remove padding
    cont = cont[padx: padx + nx, pady: pady + ny].ravel()
    return cont





def tga(x, y, data, shape, method='fd'):
    r"""
    Calculate the total gradient amplitude (TGA).
    This the same as the `3D analytic signal` of Roest et al. (1992), but we
    prefer the newer, more descriptive nomenclature suggested by Reid (2012).
    The TGA is defined as the amplitude of the gradient vector of a potential
    field :math:`T` (e.g. the magnetic total field anomaly):
    .. math::
        TGA = \sqrt{
            \left(\frac{\partial T}{\partial x}\right)^2 +
            \left(\frac{\partial T}{\partial y}\right)^2 +
            \left(\frac{\partial T}{\partial z}\right)^2 }
    .. note:: Requires gridded data.
    .. warning::
        If the data is not in SI units, the derivatives will be in
        strange units and so will the total gradient amplitude! I strongly
        recommend converting the data to SI **before** calculating the
        TGA is you need the gradient in Eotvos (use one of the unit conversion
        functions of :mod:`fatiando.utils`).
    Parameters:
    * x, y : 1D-arrays
        The x and y coordinates of the grid points
    * data : 1D-array
        The potential field at the grid points
    * shape : tuple = (nx, ny)
        The shape of the grid
    * method : string
        The method used to calculate the horizontal derivatives. Options are:
        ``'fd'`` for finite-difference (more stable) or ``'fft'`` for the Fast
        Fourier Transform. The z derivative is always calculated by FFT.
    Returns:
    * tga : 1D-array
        The amplitude of the total gradient
    References:
    Reid, A. (2012), Forgotten truths, myths and sacred cows of Potential
    Fields Geophysics - II, in SEG Technical Program Expanded Abstracts 2012,
    pp. 1-3, Society of Exploration Geophysicists.
    Roest, W., J. Verhoef, and M. Pilkington (1992), Magnetic interpretation
    using the 3-D analytic signal, GEOPHYSICS, 57(1), 116-125,
    doi:10.1190/1.1443174.
    """
    dx = derivx(x, y, data, shape, method=method)
    dy = derivy(x, y, data, shape, method=method)
    dz = derivz(x, y, data, shape)
    res = numpy.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    return res




def derivx(x, y, data, shape, order=1, method='fd'):
    """
    Calculate the derivative of a potential field in the x direction.
    .. note:: Requires gridded data.
    .. warning::
        If the data is not in SI units, the derivative will be in
        strange units! I strongly recommend converting the data to SI
        **before** calculating the derivative (use one of the unit conversion
        functions of :mod:`fatiando.utils`). This way the derivative will be in
        SI units and can be easily converted to what unit you want.
    Parameters:
    * x, y : 1D-arrays
        The x and y coordinates of the grid points
    * data : 1D-array
        The potential field at the grid points
    * shape : tuple = (nx, ny)
        The shape of the grid
    * order : int
        The order of the derivative
    * method : string
        The method used to calculate the derivatives. Options are:
        ``'fd'`` for central finite-differences (more stable) or ``'fft'``
        for the Fast Fourier Transform.
    Returns:
    * deriv : 1D-array
        The derivative
    """
    nx, ny = shape
    assert method in ['fft', 'fd'], \
        'Invalid method "{}".'.format(method)
    if method == 'fft':
        # Pad the array with the edge values to avoid instability
        padded, padx, pady = _pad_data(data, shape)
        kx, _ = _fftfreqs(x, y, shape, padded.shape)
        deriv_ft = numpy.fft.fft2(padded)*(kx*1j)**order
        deriv_pad = numpy.real(numpy.fft.ifft2(deriv_ft))
        # Remove padding from derivative
        deriv = deriv_pad[padx: padx + nx, pady: pady + ny]
    elif method == 'fd':
        datamat = data.reshape(shape)
        dx = (x.max() - x.min())/(nx - 1)
        deriv = numpy.empty_like(datamat)
        deriv[1:-1, :] = (datamat[2:, :] - datamat[:-2, :])/(2*dx)
        deriv[0, :] = deriv[1, :]
        deriv[-1, :] = deriv[-2, :]
        if order > 1:
            deriv = derivx(x, y, deriv, shape, order=order - 1, method='fd')
    return deriv.ravel()


def derivy(x, y, data, shape, order=1, method='fd'):
    """
    Calculate the derivative of a potential field in the y direction.
    .. note:: Requires gridded data.
    .. warning::
        If the data is not in SI units, the derivative will be in
        strange units! I strongly recommend converting the data to SI
        **before** calculating the derivative (use one of the unit conversion
        functions of :mod:`fatiando.utils`). This way the derivative will be in
        SI units and can be easily converted to what unit you want.
    Parameters:
    * x, y : 1D-arrays
        The x and y coordinates of the grid points
    * data : 1D-array
        The potential field at the grid points
    * shape : tuple = (nx, ny)
        The shape of the grid
    * order : int
        The order of the derivative
    * method : string
        The method used to calculate the derivatives. Options are:
        ``'fd'`` for central finite-differences (more stable) or ``'fft'``
        for the Fast Fourier Transform.
    Returns:
    * deriv : 1D-array
        The derivative
    """
    nx, ny = shape
    assert method in ['fft', 'fd'], \
        'Invalid method "{}".'.format(method)
    if method == 'fft':
        # Pad the array with the edge values to avoid instability
        padded, padx, pady = _pad_data(data, shape)
        _, ky = _fftfreqs(x, y, shape, padded.shape)
        deriv_ft = numpy.fft.fft2(padded)*(ky*1j)**order
        deriv_pad = numpy.real(numpy.fft.ifft2(deriv_ft))
        # Remove padding from derivative
        deriv = deriv_pad[padx: padx + nx, pady: pady + ny]
    elif method == 'fd':
        datamat = data.reshape(shape)
        dy = (y.max() - y.min())/(ny - 1)
        deriv = numpy.empty_like(datamat)
        deriv[:, 1:-1] = (datamat[:, 2:] - datamat[:, :-2])/(2*dy)
        deriv[:, 0] = deriv[:, 1]
        deriv[:, -1] = deriv[:, -2]
        if order > 1:
            deriv = derivy(x, y, deriv, shape, order=order - 1, method='fd')
    return deriv.ravel()


def derivz(x, y, data, shape, order=1, method='fft'):
    """
    Calculate the derivative of a potential field in the z direction.
    .. note:: Requires gridded data.
    .. warning::
        If the data is not in SI units, the derivative will be in
        strange units! I strongly recommend converting the data to SI
        **before** calculating the derivative (use one of the unit conversion
        functions of :mod:`fatiando.utils`). This way the derivative will be in
        SI units and can be easily converted to what unit you want.
    Parameters:
    * x, y : 1D-arrays
        The x and y coordinates of the grid points
    * data : 1D-array
        The potential field at the grid points
    * shape : tuple = (nx, ny)
        The shape of the grid
    * order : int
        The order of the derivative
    * method : string
        The method used to calculate the derivatives. Options are:
        ``'fft'`` for the Fast Fourier Transform.
    Returns:
    * deriv : 1D-array
        The derivative
    """
    assert method == 'fft', \
        "Invalid method '{}'".format(method)
    nx, ny = shape
    # Pad the array with the edge values to avoid instability
    padded, padx, pady = _pad_data(data, shape)
    kx, ky = _fftfreqs(x, y, shape, padded.shape)
    deriv_ft = numpy.fft.fft2(padded)*numpy.sqrt(kx**2 + ky**2)**order
    deriv = numpy.real(numpy.fft.ifft2(deriv_ft))
    # Remove padding from derivative
    return deriv[padx: padx + nx, pady: pady + ny].ravel()


def power_density_spectra(x, y, data, shape):
    r"""
    Calculates the Power Density Spectra of a 2D gridded potential field
    through the FFT:
    .. math::
        \Phi_{\Delta T}(k_x, k_y) = | F\left{\Delta T \right}(k_x, k_y) |^2
    .. note:: Requires gridded data.
    .. note:: x, y, z and height should be in meters.
    Parameters:
    * x, y : 1D-arrays
        The x and y coordinates of the grid points
    * data : 1D-array
        The potential field at the grid points
    * shape : tuple = (nx, ny)
        The shape of the grid
    Returns:
    * kx, ky : 2D-arrays
        The wavenumbers of each Power Density Spectra point
    * pds : 2D-array
        The Power Density Spectra of the data
    """
    kx, ky = _fftfreqs(x, y, shape, shape)
    pds = abs(numpy.fft.fft2(numpy.reshape(data, shape)))**2
    return kx, ky, pds


def radial_average_spectrum(kx, ky, pds, max_radius=None, ring_width=None):
    r"""
    Calculates the average of the Power Density Spectra points that falls
    inside concentric rings built around the origin of the wavenumber
    coordinate system with constant width.
    The width of the rings and the inner radius of the biggest ring can be
    changed by setting the optional parameters ring_width and max_radius,
    respectively.
    .. note:: To calculate the radially averaged power density spectra
              use the outputs of the function power_density_spectra as
              input of this one.
    Parameters:
    * kx, ky : 2D-arrays
        The wavenumbers arrays in the `x` and `y` directions
    * pds : 2D-array
        The Power Density Spectra
    * max_radius : float (optional)
        Inner radius of the biggest ring.
        By default it's set as the minimum of kx.max() and ky.max().
        Making it smaller leaves points outside of the averaging,
        and making it bigger includes points nearer to the boundaries.
    * ring_width : float (optional)
        Width of the rings.
        By default it's set as the largest value of :math:`\Delta k_x` and
        :math:`\Delta k_y`, being them the equidistances of the kx and ky
        arrays.
        Making it bigger gives more populated averages, and
        making it smaller lowers the ammount of points per ring
        (use it carefully).
    Returns:
    * k_radial : 1D-array
        Wavenumbers of each Radially Averaged Power Spectrum point.
        Also, the inner radius of the rings.
    * pds_radial : 1D array
        Radially Averaged Power Spectrum
    """
    nx, ny = pds.shape
    if max_radius is None:
        max_radius = min(kx.max(), ky.max())
    if ring_width is None:
        ring_width = max(numpy.unique(kx)[numpy.unique(kx) > 0][0],
                         numpy.unique(ky)[numpy.unique(ky) > 0][0])
    k = numpy.sqrt(kx**2 + ky**2)
    pds_radial = []
    k_radial = []
    radius_i = -1
    while True:
        radius_i += 1
        if radius_i*ring_width > max_radius:
            break
        else:
            if radius_i == 0:
                inside = k <= 0.5*ring_width
            else:
                inside = numpy.logical_and(k > (radius_i - 0.5)*ring_width,
                                           k <= (radius_i + 0.5)*ring_width)
            pds_radial.append(pds[inside].mean())
            k_radial.append(radius_i*ring_width)
    return numpy.array(k_radial), numpy.array(pds_radial)


def _pad_data(data, shape):
    n = _nextpow2(numpy.max(shape))
    nx, ny = shape
    padx = (n - nx)//2
    pady = (n - ny)//2
    padded = numpy.pad(data.reshape(shape), ((padx, padx), (pady, pady)),
                       mode='edge')
    return padded, padx, pady


def _nextpow2(i):
    buf = numpy.ceil(numpy.log(i)/numpy.log(2))
    return int(2**buf)


def _fftfreqs(x, y, shape, padshape):
    """
    Get two 2D-arrays with the wave numbers in the x and y directions.
    """
    nx, ny = shape
    dx = (x.max() - x.min())/(nx - 1)
    fx = 2*numpy.pi*numpy.fft.fftfreq(padshape[0], dx)
    dy = (y.max() - y.min())/(ny - 1)
    fy = 2*numpy.pi*numpy.fft.fftfreq(padshape[1], dy)
    return numpy.meshgrid(fy, fx)[::-1]
