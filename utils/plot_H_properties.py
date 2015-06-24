#!/usr/bin/env python
from __future__ import print_function
import argparse
import numpy as np
from statsmodels.tsa.stattools import acf
import pandas as pd
from matplotlib.pylab import figure, tight_layout, show, savefig

def get_args ():
    parser = argparse.ArgumentParser  (description='Load text-file with 2x2 MIMO matrices and plot their properties.')
    parser.add_argument ('filename', type=str,
                         help='Text file with 2x2 MIMO channel matrices.')
    parser.add_argument ('-b','--block-size',type=int,default=256,
                         help='Block-size.')
    parser.add_argument ('-g','--gain-offset',type=int,default=55,
                         help='Gain added for post-processing (in dB).')
    args = parser.parse_args ()
    return args

def get_median_CN (H):
    condition_number_dB = list ()
    for k in np.arange (H.shape[2]):
        condition_number_dB.append (20*np.log10 (np.linalg.cond (H[:,:,k])))
    print ('CN dB:',np.median (condition_number_dB))
    return np.array (condition_number_dB)

def main ():
    args = get_args ()

    data = np.genfromtxt (args.filename,dtype=np.complex128).T.reshape (2,2,-1)
    print (data.shape)

    if args.block_size is not None:
        print (data.shape[2]/args.block_size)

    print ('First matrix')
    print (data[:,:,0])
    print ('Second matrix')
    print (data[:,:,1])
    print ('Last matrix')
    print (data[:,:,-1])

    iq_power_0 = 10*np.log10 (np.power (data[0,0,:].real,2.0) + np.power (data[0,0,:].imag,2.0))
    iq_power_1 = 10*np.log10 (np.power (data[1,1,:].real,2.0) + np.power (data[1,1,:].imag,2.0))
    print ('Median IQ power:',np.median (iq_power_0)-args.gain_offset,np.median (iq_power_1)-args.gain_offset)

    cn_timeseries = get_median_CN (data)

    ax = figure ().add_subplot (111)
    ax.plot (cn_timeseries)
    ax.plot (pd.rolling_median (cn_timeseries,window=len (cn_timeseries)*0.05),color='r',lw=2.0)
    ax.grid (True)
    tight_layout ()

    r = acf (np.abs (data[0,0,200::256])**2,unbiased=True,nlags=200,fft=True)
    r_b = acf (np.abs (data[0,0,512:768])**2,unbiased=True,nlags=200,fft=True)
    ax = figure ().add_subplot (111)
    ax.plot (r)
    ax.plot (r_b)
    ax.grid (True)
    tight_layout ()

    show ()


if __name__ == '__main__':
    main ()
