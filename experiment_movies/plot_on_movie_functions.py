import cine
from pylab import *
from scipy import ndimage
from scipy.optimize import curve_fit
from numpy import *
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import scipy as sp
import pylab as P
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import os
import PIL.ImageOps
import PIL.Image
import scipy.ndimage.filters as filters
import scipy.ndimage as ndimage
import cPickle as pickle
import math
from scipy.signal import argrelextrema
import shutil
import time
import isolum_rainbow

from matplotlib.collections import PatchCollection
from functions import *


 
def plot_on_movie_circle_single_frame(c, i, time_i,  alp, output_dir, num, plotting_mag, ftf, *plotting_info): #com_data, fn = filename for com_data, plotting info has x_dat, y_dat arrays as well as coords
    '''Plots tracked data on a frame.  This will be improved soon 
    
    Parameters
    ------------
    c : cine file
        Cine file with raw data
            
    i : int
        The frame number that you want to plot from the cine file,c 
        
    time_i : float
        The time to print on the image
        
    'alp': list with floats
        alpha value (sets transparency) for both the image and the cirlce you plot
    
    output_dir: string
        the output directory for the images
    
    num: int
        the image number
    
    plotting_mag: float
        The number of times to multiply the amplitude of the displacement for drawing the circle
    
    ftf: int
        First tracked frame in the plotting info data
    
    plotting_info: list of plotting info from 'compile_x_y_traces'
        
      
    Returns:
    ------------
    time_i: float
        time of frame
    '''

    if not os.path.exists(output_dir):os.mkdir(output_dir)
    
   
    plotting_info = array(plotting_info)
 
    
    
    x_dat = plotting_info[0,0]
    y_dat = plotting_info[0,1]
    xs = plotting_info[0,2]
    ys = plotting_info[0,3]
    if len(plotting_info[0])>4:
        patch = plotting_info[0,4]
        ff = True
    else:
        ff = False
    
    num_gyros = len(xs)

    colors=zeros(num_gyros)

    
    ftf_fmf_diff =  i -ftf #difference between first movie frame and first tracked frame
    new_num = ftf_fmf_diff
    x_for_t = array(x_dat[(new_num)*num_gyros:(new_num+1)*num_gyros]) - xs  
    y_for_t = array(y_dat[new_num*num_gyros:(new_num+1)*num_gyros]) - ys
    
    
    if not ff:
        patch = []           
        for j in range(num_gyros):
       
            circ = Circle((xs[j],ys[j]), radius = sqrt((plotting_mag*x_for_t[j])**2+(plotting_mag*y_for_t[j])**2))
            patch.append(circ)

    anglez = (arctan2(y_for_t ,x_for_t))%(2*pi)
    
    x_for_t = xs + plotting_mag*x_for_t
    y_for_t = ys + plotting_mag*y_for_t
    colors = anglez
    
    
    val = 0.8 * max(c[0].astype(float).flatten())        
    minval = .10*val
    maxval = 0.6*val
    
    frame = ((clip(c[i].astype('f'), minval, maxval)-minval)/(maxval-minval))#**(1/2.2)
    
    #print 'this is the frame', frame
    
    if len(alp)>2:
        poa = alp[2]
        ax = alp[3]
    else:
        poa = True
        ax = None
        
    
    if poa: 
        x_size = 5
        y_size = 5
    
        fig=plt.figure(figsize = (x_size, y_size))
        p_ax = P.axes([0.0, 0.0, 1, 1])
    
        # print frame
        imgplot = plt.imshow(frame, cmap= cm.Greys_r, alpha = alp[0])
        p = PatchCollection(patch, cmap = 'isolum_rainbow', alpha = alp[1])
        plt.xlim(0, 600)
        plt.ylim(600,0)
        p.set_array(P.array(colors))
        p.set_clim([0, 2*pi])
        p_ax.add_collection(p)
        abc2 = p_ax.scatter(x_for_t, y_for_t, c = 'w', alpha = 1)
        #P.title('t = %0.3f s' % time_i)
        bbox_props = dict(boxstyle="round", fc="k", ec="0.2", alpha=0)
        p_ax.text(25, 35, '$t = %0.1f s$' % time_i, ha="left", va="baseline", size=32, bbox=bbox_props, color = 'w', family = 'sans-serif')
    
        p_ax.axes.get_xaxis().set_ticks([])
        p_ax.axes.get_yaxis().set_ticks([])
        print output_dir + '%04d.png' %num
    
 
        plt.savefig(output_dir + '%04d.png' %num)
        abc2.remove()
        p.remove()
        plt.close()
    else:
        if ax != None:
            p_ax = ax
            imgplot = plt.imshow(frame, cmap= cm.Greys_r, alpha =1, interpolation = 'none')
            p = PatchCollection(patch, cmap = 'isolum_rainbow', alpha = 0.85, edgecolors = 'none')
            plt.xlim(0, 600)
            plt.ylim(600,0)
            p.set_array(P.array(colors))
            p.set_clim([0, 2*pi])
            p_ax.add_collection(p)
            abc2 = p_ax.scatter(x_for_t, y_for_t, c = 'w', alpha = 1, s = 0.1)
            #P.title('t = %0.3f s' % time_i)
            bbox_props = dict(boxstyle="round", fc="k", ec="0.2", alpha=0)
            p_ax.text(25, 55, '$t = %0.1f s$' % time_i, ha="left", va="baseline", size=12, bbox=bbox_props, color = 'w', family = 'sans-serif')
    
            p_ax.axes.get_xaxis().set_ticks([])
            p_ax.axes.get_yaxis().set_ticks([])
    
    return time_i
    
def plot_on_movie_circle_single_frame_new(c, i, time_i, alp, output_dir, *plotting_info): #com_data, fn = filename for com_data, plotting info has x_dat, y_dat arrays as well as coords
    if not os.path.exists(output_dir):os.mkdir(output_dir)
    
   
    plotting_info = array(plotting_info)
    x_dat = plotting_info[0,0]
    y_dat = plotting_info[0,1]
    xs = plotting_info[0,2]
    ys = plotting_info[0,3]
    
    num_gyros = len(xs)

    colors=zeros(num_gyros)

    x_for_t = array(x_dat[i*num_gyros:(i+1)*num_gyros]) - xs  
    y_for_t = array(y_dat[i*num_gyros:(i+1)*num_gyros]) - ys
    
    patch = []           
    for j in range(num_gyros):
        circ = Circle((xs[j],ys[j]), radius = sqrt((6*x_for_t[j])**2+(6*y_for_t[j])**2))
        patch.append(circ)

    anglez = (arctan2(y_for_t ,x_for_t))%(2*pi)
    
    x_for_t = xs + 6*x_for_t
    y_for_t = ys + 6*y_for_t
    colors = anglez
    
    
    val = 0.8 * max(c[0].astype(float).flatten())        
    minval = .25*val
    maxval = 0.3*val
    frame = (clip(c[i].astype('f'), minval, maxval)-minval)/(maxval-minval)
    
    fig=plt.figure(figsize = (7,7))
    p_ax = P.axes([0.05, 0.025, 0.9, 0.95]) #axes constructor axes([left, bottom, width, height])
    imgplot = plt.imshow(frame, cmap= cm.Greys_r, alpha = alp[0])
    p = PatchCollection(patch, cmap = 'isolum_rainbow', alpha = alp[1])
    plt.xlim(0, 600)
    plt.ylim(600,0)
    p.set_array(P.array(colors))
    p.set_clim([0, 2*pi])
    p_ax.add_collection(p)
    abc2 = p_ax.scatter(x_for_t, y_for_t, c = 'White', alpha = 1)
    P.title('t = %0.1f s' % time_i)
    p_ax.axes.get_xaxis().set_ticks([])
    p_ax.axes.get_yaxis().set_ticks([])
    print output_dir + '%04d.png' %i
    plt.savefig(output_dir + '%04d.png' %i)
    abc2.remove()
    p.remove()
    plt.close()
    
    return time_i

def plot_on_movie_circle_single_frame_rec(c, i, time_i, alp, output_dir, *plotting_info): #com_data, fn = filename for com_data, plotting info has x_dat, y_dat arrays as well as coords
    if not os.path.exists(output_dir):os.mkdir(output_dir)
    
   
    plotting_info = array(plotting_info)
    x_dat = plotting_info[0,0]
    y_dat = plotting_info[0,1]
    xs = plotting_info[0,2]
    ys = plotting_info[0,3]
    
    num_gyros = len(xs)

    colors=zeros(num_gyros)

    x_for_t = array(x_dat[i*num_gyros:(i+1)*num_gyros]) - xs  
    y_for_t = array(y_dat[i*num_gyros:(i+1)*num_gyros]) - ys
    
    patch = []           
    for j in range(num_gyros):
        circ = Circle((xs[j],ys[j]), radius = sqrt((6*x_for_t[j])**2+(6*y_for_t[j])**2))
        patch.append(circ)

    anglez = (arctan2(y_for_t ,x_for_t))%(2*pi)
    
    x_for_t = xs + 6*x_for_t
    y_for_t = ys + 6*y_for_t
    colors = anglez
    
    
    val = 0.8 * max(c[0].astype(float).flatten())        
    minval = .65*val
    maxval = 1*val
    frame = (clip(c[i].astype('f'), minval, maxval)-minval)/(maxval-minval)
    
    fig=plt.figure(figsize = (7,7))
    p_ax = P.axes([0.05, 0.025, 0.9, 0.95]) #axes constructor axes([left, bottom, width, height])
    imgplot = plt.imshow(frame, cmap= cm.Greys_r, alpha = alp[0])
    p = PatchCollection(patch, cmap = 'isolum_rainbow', alpha = alp[1])
    plt.xlim(0, 800)
    plt.ylim(440,0)
    p.set_array(P.array(colors))
    p.set_clim([0, 2*pi])
    p_ax.add_collection(p)
    abc2 = p_ax.scatter(x_for_t, y_for_t, c = 'White', alpha = 1)
    P.title('t = %0.1f s' % time_i)
    p_ax.axes.get_xaxis().set_ticks([])
    p_ax.axes.get_yaxis().set_ticks([])
    print output_dir + '%04d.png' %i
    plt.savefig(output_dir + '%04d.png' %i)
    abc2.remove()
    p.remove()
    plt.close()
    
    return time_i
    
def compile_x_y_traces(base_dir):
    '''gets xy traces from tracked movie data'''
    
    filename = 'com_data.pickle'
    
    fn  =base_dir + '/com_data.pickle' #find_files_by_name(base_dir, filename, True)[0]
    

    dat = load_pickled_data(fn)
    print 'loaded ', fn

    times = (dat.T)[0]
    min_t = min(times)
    num_gyros =  len(times[where(times==min_t)])
    
    rd = fn.split('/')
    len_rd = len(rd)
    rd[0] = '/'
    rd_string = rd[:len_rd-1]
    root_dir = os.path.join(*rd_string)  + '/'
        
    '''copy the code to a directory '''
    copy_dir = root_dir + 'plotting_code'
    
    if not os.path.exists(copy_dir):os.mkdir(copy_dir)
    shutil.copy2('./'+sys.argv[0], copy_dir)
    shutil.copy2('./'+'functions.py', copy_dir)
    #copy another file

    print 'root dir', root_dir
    filename = 'x_gy_0.pickle'
    fn_array_x_gy  =find_files_by_name(root_dir, filename, True)
        
    filename = 'y_gy_0.pickle'
    fn_array_y_gy  =find_files_by_name(root_dir, filename, True)
    
    filename = 'coords_0.pickle.pickle'
    print 'filename is', filename
    fn_array_coords  =find_files_by_name(root_dir, filename, True)
    print 'fn array', fn_array_coords
   
  
    
    num_time_steps = len(times)/num_gyros
    print num_time_steps
    
    ind = array([k*num_gyros for k in range(num_time_steps)])  
    times = times[ind]
    
    window = 4
    num_gyros = len(fn_array_x_gy)     
    x_dat = zeros([num_gyros, num_time_steps]).flatten()
    y_dat = zeros([num_gyros, num_time_steps]).flatten()
    coords = []
    
    for l in range(num_gyros):
        ind2 = array([k*num_gyros+l for k in range(num_time_steps)])

        x_d = load_pickled_data(fn_array_x_gy[l])
        y_d = load_pickled_data(fn_array_y_gy[l])
        x_dat[ind2] = moving_average(x_d, window)
        y_dat[ind2] = moving_average(y_d, window)
       
        coords_s = load_pickled_data(fn_array_coords[l])
       
        coords.append(coords_s)
   
 
    coords = array(coords)
    xs = coords[:,0]
    ys = coords[:,1]
    

    print len(x_dat)
    print num_time_steps
    print len(x_dat)/num_time_steps
    
    return [x_dat, y_dat, xs, ys], times
         