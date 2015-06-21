#!/usr/bin/env python
"""This script generates the waveform and save it in binary format to
a file

"""
import argparse, sys
import numpy as np
from gnuradio import gr, digital, blocks, filter
import scipy.signal as signal
from scipy.fftpack import fft, fftshift
from matplotlib.pyplot import figure, tight_layout, show

def get_args ():
    parser = argparse.ArgumentParser (description='Generate waveform.')
    parser.add_argument('-n','--number-of-symbols', type=int,
                        default=128, help='Number of "data" symbols to generate (assuming the underlying modulation.')
    parser.add_argument('-s','--samples-per-symbol', type=int,
                        default=4, help='Number of samples per symbol (default is 4).')
    parser.add_argument('-a','--rolloff-factor', type=float,
                        default=0.35, help='Excess bandwidth or roll-off factor (alpha, default is 0.35).')
    parser.add_argument('-K', type=int,
                        default=12, help='Length of the pulse shaping filter in symbol units (default is 12).')
    parser.add_argument('-m', '--modulation-order',type=int,
                        default=4, help='Modulation order (default is 4 e.g. QPSK).')
    parser.add_argument('--seed', type=int,
                        default=42, help='Seed of the random number generator used to generate data.')
    parser.add_argument('-p','--periodicity', type=int,
                        help='Periodicity of the underlying symbol sequence.')
    parser.add_argument('-l','--load-symbols', type=str,nargs='+',
                        help='Load symbols directly from a binary file. In this case, the periodicity and number-of-samples options are ignored.')
    parser.add_argument('-i','--index', type=int, default=0,
                        help='Stream index (default is 0).')
    parser.add_argument('--use-arb', action='store_true',
                        help='Use polyphase filterbank for pulse-shaping.')
    parser.add_argument('--plot', action='store_true',
                        help='Display plots.')
    args = parser.parse_args ()
    return args

class top_block (gr.top_block):
    def __init__ (self,number_of_symbols,samples_per_symbol,alpha,K,
                  modulation_order=4,
                  seed=42,periodicity=None,load_symbols=None,index=0,
                  use_arb=False):
        gr.top_block.__init__(self)

        # FIXME For preamble: see simulations/baseband_model_pulse_shaping.py

        if load_symbols is not None:
            # Add dummy symbols to initialize the filter
            if use_arb:
                zero_padding = np.ones (K*samples_per_symbol,dtype=np.complex64)
            else:
                zero_padding = np.ones (K,dtype=np.complex64)
            symbols = np.fromfile (load_symbols[0],dtype=np.complex64,count=-1)
            print len (symbols),symbols # Debug
            self._symbols = blocks.vector_source_c (np.concatenate ([symbols,zero_padding]).astype (np.complex),repeat=False)
        else:
            if modulation_order==2:
                self._constellation = digital.constellation_bpsk ()
            else:
                # Setup constellation for data part: QPSK for now per default
                self._constellation = digital.constellation_qpsk ()
            print 'Constellation:', self._constellation.points (), self._constellation.arity ()  # Debug
            self._to_symbols = digital.chunks_to_symbols_ic (self._constellation.points ())

            # Generate data: assuming a QPSK constellation
            print 'RNG seed:', seed
            np.random.seed (seed)

            # Add dummy symbols to initialize the filter
            if use_arb:
                zero_padding = np.ones (K*samples_per_symbol,dtype=int)+self._constellation.arity ()
            else:
                zero_padding = np.ones (K,dtype=int)+self._constellation.arity ()
            if periodicity is not None:
                data = np.tile (np.random.randint (0,self._constellation.arity (),periodicity) % self._constellation.arity (),(1,number_of_symbols//periodicity)).flatten ()
            else:
                data = np.random.randint (0,self._constellation.arity (),number_of_symbols)
            print len (data),data  # Debug
            self._data = blocks.vector_source_i (np.concatenate ([data,zero_padding]),repeat=False)

        # Pulse shaping
        if use_arb:
            nfilts = 32
            ntaps = nfilts * K * samples_per_symbol  # make nfilts filters
            # of ntaps each. First two parameters are: gain and sampling
            # rate based on nfilts filters in resampler (and assuming
            # normalized frequencies)
            self._tx_rrc_taps = filter.firdes.root_raised_cosine (nfilts,nfilts,1.0,alpha,ntaps)
            rrc_tx_filter = filter.pfb_arb_resampler_ccf (samples_per_symbol,
                                                          self._tx_rrc_taps)
        else:
            ntaps = K * samples_per_symbol
            self._tx_rrc_taps = filter.firdes.root_raised_cosine (samples_per_symbol, samples_per_symbol, 1.0,alpha, ntaps)
            rrc_tx_filter = filter.interp_fir_filter_ccf (samples_per_symbol,
                                                          self._tx_rrc_taps)
        print 'Number of taps for pulse-shaping:', ntaps

        self._sink = blocks.file_sink (gr.sizeof_gr_complex,'s_pulse_shaped_signal_sps{:d}_{:d}.bin'.format (samples_per_symbol,index))

        # Debug
        self._modulation_out = blocks.vector_sink_c ()
        self._pulse_shaping_out = blocks.vector_sink_c ()

        # Connect
        if load_symbols is not None:
            self.connect (self._symbols,rrc_tx_filter,self._sink)
            self.connect (self._symbols,self._modulation_out)
        else:
            self.connect (self._data,self._to_symbols,rrc_tx_filter,self._sink)
            self.connect (self._to_symbols,self._modulation_out)
        self.connect (rrc_tx_filter,self._pulse_shaping_out)

if __name__ == '__main__':
    args = get_args ()

    tb = top_block (args.number_of_symbols,args.samples_per_symbol,args.rolloff_factor,args.K,
                    args.modulation_order,
                    seed=args.seed,periodicity=args.periodicity,load_symbols=args.load_symbols,index=args.index,
                    use_arb=args.use_arb)
    tb.start ()
    tb.wait ()

    # Print outcome for verification purpose: note that for the pulse-shaper output, the first K*number_of_samples are lost because of filter delay. This corresponds to the first K symbols
    if args.use_arb:
        stop = -1
    else:
        stop = -args.K
    modulation_out = np.array (tb._modulation_out.data ()[:stop])
    # assert (len (modulation_out) == args.number_of_symbols-args.K)

    if args.use_arb:
        start = (args.samples_per_symbol/2)*len (tb._tx_rrc_taps)//32-1
        stop = start + args.number_of_symbols*args.samples_per_symbol
    else:
        start = len (tb._tx_rrc_taps)//2
        stop = -start

    ps_out = np.array (tb._pulse_shaping_out.data ()[start:stop])
    print 'Pulse-shaper output length:',len (tb._pulse_shaping_out.data ())

    if not args.plot:
        sys.exit (0)

    f = figure ()
    ax = f.add_subplot (211)
    ax.plot (modulation_out.real,modulation_out.imag,'.')
    ax.grid (True)
    ax = f.add_subplot (212)
    ax.plot (modulation_out.real)
    ax.plot (modulation_out.imag)
    ax.grid (True)
    tight_layout ()

    f = figure ()
    ax = f.add_subplot (211)
    ax.plot (ps_out.real,ps_out.imag,'.')
    ax.grid (True)
    ax = f.add_subplot (212)
    ax.plot (ps_out.real)
    ax.plot (ps_out.imag)
    ax.grid (True)
    tight_layout ()

    f = figure ()
    ax = f.add_subplot (111)
    w,h = signal.freqz (ps_out,worN=4096,whole=True)
    ax.plot (w,20*np.log10 (np.abs (h)),color='m')
    ax.grid (True)
    tight_layout ()

    show ()
