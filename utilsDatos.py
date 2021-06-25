# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 13:16:42 2021
@author: Gabriel R. Gelpi
"""
import netCDF4     
import numpy as np
import time




def f_txt2nc(long,lati,dato,name_f='nova2',titulo = 'Mis datos',descr = "se guardan las coordenadas y algun otro dato de interes",
             historia = "Fecha de cracion hoy", sourc = "Tutorial: https://github.com/Unidata/netcdf4-python/blob/master/examples/writing_netCDF.ipynb"):
    
    l_long = len(long)
    l_lat = len(lati)
    n_la,n_lo = dato.shape
    
    try: ncfile.close()  # revisamos que no haya nada abierto
    except: pass
    ncfile = netCDF4.Dataset(name_f+'.nc',mode='w',format='NETCDF4_CLASSIC') 
    
    
    # Primero se crean las dimensiones de las variables a utilizar    
    lat_dim = ncfile.createDimension('lat', l_lat)     # latitude axis
    lon_dim = ncfile.createDimension('lon', l_long)    # longitude axis
    d_dim = ncfile.createDimension('dato', None) # unlimited axis (can be appended to).
    
    
    # se crean las variables donde se almacenaran los datos
    lat = ncfile.createVariable('lat', np.float32, ('lat',))
    lat.units = 'degrees_north'
    lat.long_name = 'latitude'
    lon = ncfile.createVariable('lon', np.float32, ('lon',))
    lon.units = 'degrees_east'
    lon.long_name = 'longitude'
    d = ncfile.createVariable('dato', np.float64, ('lat','lon'))
    d.units = 'mgal'
    d.long_name = 'time'
    
    # asignamos algunos atributos al archivo:
    ncfile.title= titulo
    ncfile.description = descr
    ncfile.history = historia #+ d.ctime(d.time())
    ncfile.source = sourc    
        
    # Ahora si se guarda la info   
    lat[:] = lati
    lon[:] = long
    d[:,:] = dato 

    print(ncfile)
    # cerramos el archivo.
    ncfile.close(); print('Dataset is closed!')
    
    
def obt_coord(x,y,d):
    
    """
    funcion que devuelve la grilla de coordenadas para el archivo .nc de GMT.
    x: longitud
    y: latitud
    xc: grilla de coordenadas para la longitud
    yc: grilla de coordenadas para la latitud
    d: dato (anomalias de gravedad, topogorafia, etc...)
    """
    xc = np.empty_like(d)
    yc = np.empty_like(d)
    nf,nc = np.shape(d)
    
    xc = np.tile(x,(nf,1))
    yc = np.transpose([y]*nc)
    
    return xc,yc



