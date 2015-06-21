#!/usr/bin/env python
"""This script loads a waveform from binary file and pass it through a
channel

"""
from __future__ import print_function
import argparse, sys
import numpy as np
from gnuradio import gr, blocks, channels, filter
from matplotlib.pyplot import figure, tight_layout, show

def get_args ():
    parser = argparse.ArgumentParser (description='Pass waveform through channel.')
    parser.add_argument('filename', type=str, nargs='+',
                        help='File to load for the waveform.')
    parser.add_argument('-f','--frequency-offset', type=float, default=0.0,
                        help='Frequency offset (default value is 0.0).')
    parser.add_argument('-t','--timing-offset', type=float, default=1.0,
                        help='Timing offset (default value is 1.0).')
    parser.add_argument('-v','--noise-voltage', type=float, default=0.0,
                        help='Noise voltage (default value is 0.0).')
    parser.add_argument('-d','--delay', type=int, default=0,
                        help='Arbitrary sample delay (default value is 0).')
    parser.add_argument('-m','--mimo', action='store_true',
                        help='Implement a canonical MIMO channel (only 2x2 supported for now).')
    parser.add_argument('-s','--samples-per-symbol', type=int,
                        default=4, help='Number of samples per symbol (default is 4).')
    parser.add_argument('-a','--rolloff-factor', type=float,
                        default=0.35, help='Excess bandwidth or roll-off factor (alpha, default is 0.35).')
    parser.add_argument('-K', type=int,
                        default=12, help='Length of the pulse shaping filter in symbol units (default is 12).')
    parser.add_argument('--plot', action='store_true',
                        help='Display plots.')
    args = parser.parse_args ()
    return args

class top_block (gr.top_block):
    def __init__ (self,filename,frequency_offset,timing_offset,noise_voltage,do_mimo,channel_delay=0,
                  K=12,samples_per_symbol=4,alpha=0.35):
        gr.top_block.__init__(self)

        self._src = list ()
        channel = list ()
        self._dst = list ()
        self._tx_waveform = list ()
        self._rx_waveform = list ()
        self._rrc_filter = list ()

        ntaps = K * samples_per_symbol
        self._tx_rrc_taps = filter.firdes.root_raised_cosine (samples_per_symbol, samples_per_symbol, 1.0, alpha, ntaps)
        print ('Number of taps for pulse-shaping:', ntaps)

        for k in np.arange (len (filename)):
            # Load binary file
            print ('Load file {:s}'.format (filename[k]))
            self._src.append (blocks.file_source (gr.sizeof_gr_complex,filename[k]))

            self._rrc_filter.append (filter.interp_fir_filter_ccf (samples_per_symbol,self._tx_rrc_taps))

            # Channel: see https://gnuradio.org/doc/doxygen/page_channels.html
            channel.append (channels.channel_model (frequency_offset=frequency_offset,
                                                    epsilon=timing_offset,
                                                    noise_voltage=noise_voltage,
                                                    taps=[1,0]))
            print (channel[-1].taps ())

            self._dst.append (blocks.file_sink (gr.sizeof_gr_complex,'rx_channel_sps{:d}_'.format (samples_per_symbol)+filename[k]))

            # Debug outputs
            self._tx_waveform.append (blocks.vector_sink_c ())
            self._rx_waveform.append (blocks.vector_sink_c ())

        if do_mimo:
            assert len (filename) == 2
            print ('2x2 MIMO channel')
            self.connect (self._src[0],self._rrc_filter[0])
            self.connect (self._src[1],self._rrc_filter[1])

            diag_term_01 = blocks.multiply_const_cc (1j)
            diag_term_10 = blocks.multiply_const_cc (1j)
            add_01 = blocks.add_cc ()
            add_10 = blocks.add_cc ()
            # 0 + j*1
            self.connect (self._rrc_filter[0],(add_01,0))
            self.connect (self._rrc_filter[1],diag_term_10,(add_01,1))
            # j*0 + 1
            self.connect (self._rrc_filter[0],diag_term_01,(add_10,0))
            self.connect (self._rrc_filter[1],(add_10,1))

            self.connect (add_01,blocks.delay (gr.sizeof_gr_complex,3+channel_delay),channel[0],self._dst[0])
            self.connect (add_10,blocks.delay (gr.sizeof_gr_complex,3+channel_delay),channel[1],self._dst[1])

        else:
            print ('Parallel channels')
            for k in np.arange (len (filename)):
                # Connect: the channel_delay is added to compensate for the
                # effect of the channel block (3 samples earlier)
                self.connect (self._src[k],self._rrc_filter[k],blocks.delay (gr.sizeof_gr_complex,3+channel_delay),channel[k],self._dst[k])
        # Debug output
        for k in np.arange (len (filename)):
            self.connect (self._rrc_filter[k],self._tx_waveform[k])
            self.connect (channel[k],self._rx_waveform[k])
        

if __name__ == '__main__':
    args = get_args ()

    tb = top_block (args.filename,args.frequency_offset,args.timing_offset,args.noise_voltage,args.mimo,args.delay,
                    args.K,args.samples_per_symbol,args.rolloff_factor)
    tb.start ()
    tb.wait ()

    if not args.plot:
        sys.exit (0)

    tx_waveform = np.array (tb._tx_waveform[0].data ())
    print ('Waveform length:',len (tx_waveform))

    rx_waveform = np.array (tb._rx_waveform[0].data ())
    print ('Waveform length after channel:',len (tx_waveform))

    f = figure ()
    ax = f.add_subplot (211)
    ax.plot (tx_waveform.real,tx_waveform.imag,'.')
    ax.plot (rx_waveform.real,rx_waveform.imag,'x',color='Magenta')
    ax.grid (True)
    ax = f.add_subplot (212)
    ax.plot (tx_waveform.real)
    ax.plot (rx_waveform.real,color='LightBlue',ls='--')
    ax.plot (tx_waveform.imag)
    ax.plot (rx_waveform.imag,color='LightGreen',ls='--')
    ax.grid (True)
    tight_layout ()

    show ()
