#!/usr/bin/env python
"""Generate symbols of the measurement waveform. Symbols can then be
loaded by generate_waveform.py. Currently a QPSK modulation is assumed
for the data symbols

"""
from __future__ import print_function
import argparse
import numpy as np
from measurement_waveform import measurementSymbols
from matplotlib.pyplot import figure, tight_layout, show

def get_args ():
    parser = argparse.ArgumentParser  (description='Generate symbols of the measurement waveform.',
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument ('-n','--number-of-symbols', type=int,
                         default=32, help='Number of "data" symbols to generate (assuming the underlying modulation.')
    parser.add_argument ('-p','--periodicity', type=int,
                         help='Periodicity of the underlying symbol payload sequence.')
    parser.add_argument ('-r','--repeat-symbols-sequence',type=int,
                         help='Repeat the complete symbol sequence')
    parser.add_argument ('--disable-preamble',action='store_true',
                         help='Disable the generation of the preamble.')
    parser.add_argument ('--disable-pilot',action='store_true',
                         help='Disable the generation of the pilot.')
    parser.add_argument ('--single-preamble',action='store_true',
                         help='Transmit preamble only on first stream.')
    parser.add_argument ('-m','--mimo-streams',default=1,type=int,
                         help='Number of MIMO streams to generate.')
    parser.add_argument ('--preamble-seq-length',default=8,type=int,
                         help='Number of repetitions of the pilot root sequence.')
    parser.add_argument ('--preamble-root-length',default=64,type=int,
                         help='Length (in symbols) of the pilot root sequence.')
    parser.add_argument ('--pilot-seq-length',default=256,type=int,
                         help='Number of repetitions of the pilot root sequence.')
    parser.add_argument ('--pilot-root-length',default=8,type=int,
                         help='Length (in symbols) of the pilot root sequence.')
    parser.add_argument ('--seed', type=int,
                         default=42, help='Seed of the random number generator used to generate data.')
    parser.add_argument ('--plot', action='store_true',
                         help='Display plots.')
    args = parser.parse_args ()
    return args

def main ():
    args = get_args ()

    mw = measurementSymbols (payload_length=args.number_of_symbols,
                             mimo_streams=args.mimo_streams,
                             periodicity=args.periodicity,
                             repeat_symbols=args.repeat_symbols_sequence,
                             preamble_seq_length=args.preamble_seq_length,preamble_root_length=args.preamble_root_length,
                             pilot_seq_length=args.pilot_seq_length,pilot_root_length=args.pilot_root_length,
                             seed=args.seed)
    mw.generate_symbols (with_preamble=not args.disable_preamble,with_pilot=not args.disable_pilot,single_preamble=args.single_preamble)
    print ('[Debug] data', mw.data[:10])
    print ('Symbol sequence length:', mw.symbols.shape[1])
    mw.to_file ()

    if args.plot:
        if args.mimo_streams == 2:
            f = figure ()
            ax = f.add_subplot (211)
            ax.plot (mw.symbols[0].real,mw.symbols[0].imag,'xm')
            ax.set_ylabel ('Quadrature')
            ax.grid (True)
            ax = f.add_subplot (212)
            ax.plot (mw.symbols[1].real,mw.symbols[1].imag,'xm')
            ax.set_xlabel ('In-phase')
            ax.set_ylabel ('Quadrature')
            tight_layout ()

            f = figure ()
            ax = f.add_subplot (211)
            ax.plot (mw.symbols[0].real)
            ax.plot (mw.symbols[0].imag,'m')
            ax = f.add_subplot (212)
            ax.plot (mw.symbols[1].real)
            ax.plot (mw.symbols[1].imag,'m')
            tight_layout ()
        else:
            f = figure ()
            ax = f.add_subplot (211)
            ax.plot (mw.symbols[0].real,mw.symbols[0].imag,'xm')
            ax.set_ylabel ('Quadrature')
            ax.grid (True)
            ax = f.add_subplot (212)
            ax.plot (mw.symbols[0].real)
            ax.plot (mw.symbols[0].imag,'m')
        

        show ()

if __name__ == '__main__':
    main ()
