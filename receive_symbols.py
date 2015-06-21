#!/usr/bin/env python
"""Receive (decode) symbols of a measurement waveform. Symbols have
been demodulated by receive_waveform.py

"""
from __future__ import print_function
import argparse
import numpy as np
from matplotlib.pyplot import figure, tight_layout, show
from measurement_waveform import measurementSymbols, symbolReceiver

def get_args ():
    parser = argparse.ArgumentParser  (description='Receive symbols of the measurement waveform.')
    parser.add_argument ('filename', type=str, nargs='+',
                         help='Binary file with received symbols.')
    parser.add_argument ('-n','--number-of-symbols', type=int,
                         default=32, help='Number of "data" symbols to generate (assuming the underlying modulation.')
    parser.add_argument ('-p','--periodicity', type=int,
                         help='Periodicity of the underlying symbol sequence.')
    parser.add_argument ('-s','--sync', action='store_true',
                         help='Run preamble synchronization on the received symbols sequence.')
    parser.add_argument ('--skip-pilot', action='store_true',
                         help='Skip the pilot sequence (assuming it is present in the received symbols.')
    parser.add_argument ('-c','--correct-phase', action='store_true',
                         help='Correct residual phase offset.')
    parser.add_argument ('-i','--pilot-id', nargs='+',default=[-1],type=int,
                         help='Pilot id list (default is [-1]')
    parser.add_argument ('-g','--gain',default=1,type=int,
                         help='Gain for frame synchronization (number of repetitions of the mask, MIMO only).')
    parser.add_argument ('--check',default=3,type=int,
                         help='How many times must the synchronization with the mask occur (default is 3, MIMO only).')
    parser.add_argument ('--pilot-seq-length',default=256,type=int,
                         help='Number of repetitions of the pilot root sequence.')
    parser.add_argument ('--pilot-root-length',default=8,type=int,
                         help='Length (in symbols) of the pilot root sequence.')
    parser.add_argument ('--seed', type=int,
                         default=42, help='Seed of the random number generator used to generate data.')
    parser.add_argument ('--save-channels', type=str,
                         help='Export channels to file.')
    parser.add_argument ('-f','--frames', type=int,
                         help='Process only that number of frames.')
    parser.add_argument ('--plot', action='store_true',
                         help='Display plots.')
    args = parser.parse_args ()
    return args

def plot_received_symbols (symbols,valid_ratio=0.8):
    """Plot the received symbols

    symbols    : a numpy array containing the received symbols
    valid_ratio: because a rotation is expected at the beginning,
                 ratio of symbols to plot where it is assumed that
                 the rotation is compensated

    Mostly to verify that phase and timing issues have been sorted out

    """
    ax = figure ().add_subplot (111)
    ax.plot (symbols.real,symbols.imag,'xm')
    ax.plot (symbols.real[-len (symbols)*valid_ratio:],symbols.imag[-len (symbols)*valid_ratio:],'.c')
    ax.set_xlabel ('In-phase')
    ax.set_xlabel ('Quadrature')
    ax.grid (True)
    tight_layout ()

    # ax = figure ().add_subplot (111)
    # ax.plot (symbols.real,'m')
    # ax.plot (symbols.imag,'c')
    # ax.grid (True)
    # tight_layout ()

def main ():
    args = get_args ()

    sym_receiver = symbolReceiver (payload_length=args.number_of_symbols,periodicity=args.periodicity,
                                   pilot_seq_length=args.pilot_seq_length,pilot_root_length=args.pilot_root_length,
                                   seed=args.seed)
    n_streams = len (args.filename)
    assert n_streams <= 2
    # Load symbols
    if n_streams == 1:
        sym_receiver.load_stream (args.filename[0])
    else:
        sym_receiver.load_multiple_streams (args.filename)

    if args.sync:
        if n_streams == 1:
            sym_receiver.siso_sync_and_decode (gain=args.gain,skip_pilot=args.skip_pilot,correct_phase=args.correct_phase,pilot_id=args.pilot_id)
        if n_streams == 2:
            if args.frames is not None:
                args.frames += -1
            channels = sym_receiver.mimo_2x2_sync_and_decode (gain=args.gain,check_needed=args.check,correct_phase=args.correct_phase,pilot_id=args.pilot_id,save_channels=args.save_channels,count=args.frames)
            if args.save_channels is not None and channels is not None:
                print ('Export {:d} blocks of {:d} channels'.format (len (channels),sym_receiver.pilot_seq_length))
                with file (args.save_channels,'w') as f:
                    for k,block in enumerate (channels):
                        f.write ('# Block {:d}\n'.format (k))
                        for l in np.arange (block.shape[2]):
                            # Use genfromtxt to recover the data
                            np.savetxt (f,block[:,:,l].reshape (-1,4),fmt=['%.15e%+.15ej']*4)
    # In case of a periodic waveform: check property
    if args.periodicity:
        sym_receiver.check_periodicity ()

    if args.plot:
        plot_received_symbols (sym_receiver.rx_symbols)
        show ()

if __name__ == '__main__':
    main ()
