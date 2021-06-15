# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 09:30:49 2021

@author: Gabriel R. Gelpi
"""
from __future__ import absolute_import, division
#from future.builtins import range
#import math

import numpy
import scipy.sparse
#import scipy.sparse.linalg
#import scipy.misc
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

#==========================================================
SI2MGAL = 100000.0
G = 0.00000000006673


def si2mgal(value):
    """
    Convert a value from SI units to mGal.
    Parameters:
    * value : number or array
        The value in SI
    Returns:
    * value : number or array
        The value in mGal
    """
    return value * SI2MGAL


def mgal2si(value):
    """
    Convert a value from mGal to SI units.
    Parameters:
    * value : number or array
        The value in mGal
    Returns:
    * value : number or array
        The value in SI
    """
    return value / SI2MGAL

#==============================
def plot_1x1(data,name='dato',interp=None,mapa_colo='seismic',graf=0,n_plot='una_fig'):
    """
    Grafico de una imágen.
    Parameters:
    * data: dato que se quiere graficar.
    * name: nombre del titulo de la figura.
    * interp: interpolacion de la imagen. None es ninguna. Para agregar se puede elegir: 
    Ej 'bicubic','nearest','gaussian', etc...
    * mapa_colo: Paleta de colores. Ej: 'gray','jet','bones',etc..     
    * graf: 0 guarda la figura, distinto de 0 no la guarda.
    * una_fig: Nombre que se quiere dar al grafico
    """
    fig = plt.figure(figsize=(6,7))

    ax = fig.add_subplot(1, 1, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.imshow(data, aspect='auto', interpolation=interp, 
               cmap=mapa_colo)
    plt.title(name)
    plt.colorbar()
    
    if graf == 1:
        fig.savefig(n_plot+'.pdf',dpi=200,bbox_inches='tight')
    plt.show()
    
    return print('nice plot!')


#==============================


def plot_1x2(d,dp1,name1='data',name2='data2',interp=None, mapa_colo='seismic',
             graf=0,n_plot='dos_fig'):
    
    fig,ax = plt.subplots(1,2,figsize=(10,6))
    
    for i in range(0,2): ax[i].set_xticks([])
    for i in range(0,2): ax[i].set_yticks([])
    
    im1 = ax[0].imshow(d,aspect='auto',interpolation=interp,cmap=mapa_colo)
    ax[0].set_title(name1)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, ax=ax[0],cax=cax)
    
    im2 = ax[1].imshow(dp1,aspect='auto',interpolation=interp,cmap=mapa_colo)
    ax[1].set_title(name2)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, ax=ax[1],cax=cax)
    
    if graf == 1:
        fig.savefig(n_plot+'.pdf',dpi=200,bbox_inches='tight')
    
    
    return print('nice plot!')

#======================================
def plot_1x3(d,dp1,dp2,name1='data',name2='data2',name3='data3',interp=None, 
             mapa_colo='seismic',graf=0, n_plot='tres_fig'):
    
    fig,ax = plt.subplots(1,3,figsize=(15,6))
    
    for i in range(0,3): ax[i].set_xticks([])
    for i in range(0,3): ax[i].set_yticks([])

    im1 = ax[0].imshow(d,aspect='auto',interpolation=interp,cmap=mapa_colo)
    ax[0].set_title(name1)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, ax=ax[0],cax=cax)
    
    im2 = ax[1].imshow(dp1,aspect='auto',interpolation=interp,cmap=mapa_colo)
    ax[1].set_title(name2)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, ax=ax[1],cax=cax)
    
    im3 = ax[2].imshow(dp2,aspect='auto',interpolation=interp,cmap=mapa_colo)
    ax[2].set_title(name3)
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im3, ax=ax[2],cax=cax)
    
    if graf == 1:
        fig.savefig(n_plot+'.pdf',dpi=200,bbox_inches='tight')

    
        
    
    #plt.title(name)
    return print('nice plot!')
    
    
    
    
    