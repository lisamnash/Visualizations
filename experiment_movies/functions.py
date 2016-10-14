import cine
from pylab import *
from scipy import ndimage
from scipy.optimize import curve_fit
from numpy import *
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
import itertools
import time
import pylab as P
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import sys


def load_pickled_data(root_dir, filename = -1):
    '''loads data from pickle file .

    Parameters
    ----------
    root_dir : string
        Directory in which file is saved
    filename : string 
        name of file

    Returns
    ----------
    data : any python object
        data from pickled file
        '''
    if filename ==-1:
        tot_path = root_dir
    else:
        tot_path = root_dir + '/' + filename

    try :
        of = open(tot_path, 'rb')
        data = pickle.load(of)
    except Exception:
        data = 0
        print 'file not found', tot_path
        sys.exit()
    return data

def dump_text_data(output_dir, filename, data):
    con = open(output_dir + '/' + filename + '.csv', "wb")
    con_len = len(data)
    data = array(data)
    print data[0]
    print data[con_len-1]

    for i in range(con_len):
        for j in range(len(data[0])):
            #print i, j
            #print data[i,j]
            con.write(str(data[i,j]) + ' ,') 
        con.write('\n')
    con.close()

def dump_pickled_data(output_dir, filename, data):
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    of = open(output_dir + '/'+filename + '.pickle', 'wb')
    pickle.dump(data, of, pickle.HIGHEST_PROTOCOL)
    of.close()
    
def func(x, xmax, tau, f, phi):
    return (xmax*exp(-1/tau * x)*sin(2*pi* f *x + phi))
   
   
def shift(yvals):
    y_mean = mean(yvals)
    yvals = yvals - y_mean
    return yvals


def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx
   
        

