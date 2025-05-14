#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import minimum_filter


def read_and_filter(fname, fmin = -32, fmax = 64):
    '''
    Reads file and filters data
    
    Input:
    fname = input file, assumed to be text separated by ;
    fmin = minimum valid value
    fmax = maximum valid value

    Output:
    data = numpy array of integers
    '''

    #read file
    with open(fname, 'r') as file:
        data = np.array([line.strip().split(";") for line in file.readlines()]).astype(int)

    #filter data
        data = np.where((data < fmin) | (data > fmax), np.nan, data)

    return data

def plot_radar(data, ax, vmin=-32, vmax=64,plot_zero=True):
    '''
    plotting the radar on a matplotlib figure axis
    Input:
        data : 2d numpy array
        ax: matplotlib figure axis
        vmin, vmax: min and max of the scale
        plot_zero: whether to plot the lowest value
    Output: figure axis
    '''
    if (plot_zero):
        
        ax.imshow(data, vmin=vmin, vmax=vmax, cmap='viridis')

    else:

        fdata = np.where(data == vmin,np.nan,data)
        ax.imshow(fdata, vmin=vmin,vmax = vmax, cmap='viridis')

    return ax

def simple_composite(img1, img2, smooth=False):
    '''
    Very simple solution: each img fills locations where it is unique
    locations with data from both images is averaged
    
    Input: 
        img1, img2: n-D numpy arrays that are the same size
        smooth: option for very slightly filtering to remove edges

    
    Output: 
        n-D numpy array with merged values
    '''
    fimg = np.nanmean( np.array([ img1, img2 ]), axis=0 )
    if smooth:
        return minimum_filter(fimg, size=2)
    else:
        return fimg


def fill_missing(data, method='linear'):
    '''
    Interpolates missing values (NaNs) using scipy.interpolate.griddata 

    Input: 
        data: 2D-numpy array
        method: interpolation method to use ('linear’, ‘nearest’, ‘cubic’)
        
    Output: 
        data: Same array filled

    '''
    # Find missing
    missing_indices = np.isnan(data)

    x, y = np.mgrid[0:data.shape[0], 0:data.shape[1]]

    valid_x = x[~missing_indices]
    valid_y = y[~missing_indices]
    valid_values = data[~missing_indices]

    # Interpolate missing values using linear interpolation
    interpolated_values = griddata((valid_x, valid_y), valid_values, (x[missing_indices], y[missing_indices]), method=method)

    data[missing_indices] = interpolated_values

    return data

if (__name__ == "__main__"):

    #read and filter raw data
    dleft = read_and_filter('left.csv')
    dright = read_and_filter('right.csv')
    
    #plot raw data
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    ax[0] = plot_radar(dleft, ax[0])
    ax[1] = plot_radar(dright, ax[1])
    fig.savefig('raw_left_right.png')

    #create VERY simple composite
    comp = simple_composite(dleft,dright,smooth=True)

    #plot
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize = (8,8))
    axs = plot_radar(comp,axs)
    fig.savefig('simple_composite_nofill.png')

    #fill missing data
    fcomp = fill_missing(comp, method ='linear') #infill
    fcomp = fill_missing(comp, method='nearest') #fill edges

    #final plot
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize = (8,8))
    axs = plot_radar(fcomp,axs)
    fig.savefig('simple_composite_fill.png')

    #output final data
    np.savetxt('final.csv', fcomp, delimiter=';',fmt='%d')



