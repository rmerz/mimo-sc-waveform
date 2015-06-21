#!/usr/bin/env python
"""This script receives a pulse-shaped waveform and attempts to decode
it

"""
from __future__ import print_function
import argparse, sys
import numpy as np
from gnuradio import gr, digital, blocks, filter, channels
from matplotlib.pyplot import figure, tight_layout, show

def get_args ():
    parser = argparse.ArgumentParser (description='Receive a pulse-shaped waveform.')
    parser.add_argument ('filename', type=str, nargs='+',
                         help='File to load for the waveform.')
    parser.add_argument ('-n','--number-of-symbols', type=int,
                         default=1024, help='Number of "data" symbols expected per block.')
    parser.add_argument ('-s','--samples-per-symbol', type=int,
                         default=4, help='Number of samples per symbol (default is 4).')
    parser.add_argument ('-a','--rolloff-factor', type=float,
                         default=0.35, help='Excess bandwidth or roll-off factor (alpha, default is 0.25).')
    parser.add_argument ('-K', type=int,
                         default=12, help='Length of the pulse shaping filter in symbol units (default is 8).')
    parser.add_argument('-m', '--modulation-order',type=int,
                        default=4, help='Modulation order (default is 4 e.g. QPSK).')
    parser.add_argument ('--seed', type=int,
                         default=42, help='Seed of the random number generator used to generate data.')
    parser.add_argument ('-F','--to-file', type=str, nargs='+',
                         help='Save output of matched filter to binary file.')
    parser.add_argument ('-f','--correct-frequency-offset', action='store_true',
                         help='Correct frequency offset.')
    parser.add_argument ('-o','--frequency-offset', type=float, default=0,
                         help='Initial frequency offset correction.')
    parser.add_argument ('--fll-ntaps-factor', type=float, default=1.5,
                         help='Multiplying factor for the number of taps for the FLL filter: factor * K * samples_per_symbol.')
    parser.add_argument ('-g','--gain-dB', type=float, default=0,
                         help='Initial gain added to the received signal (in dB, default is 0).')
    parser.add_argument ('-t','--correct-timing-offset', action='store_true',
                         help='Correct timing offset.')
    parser.add_argument ('--plot-offset-correction', action='store_true',
                         help='Display frequency and/or timing offset debug output plots.')
    parser.add_argument ('--plot-mf-output', action='store_true',
                         help='Display matched-filter output plots.')
    args = parser.parse_args ()
    return args

class top_block (gr.top_block):
    def __init__ (self,filename,
                  samples_per_symbol,alpha,K,
                  modulation_order=4,
                  correct_frequency_offset=False,correct_timing_offset=False,
                  fll_ntaps_factor=1.5,frequency_offset=0,
                  gain_dB=0,
                  seed=42,to_file=None):
        gr.top_block.__init__(self)
        print ('Roll-off:', alpha)

        print ('Filenames:',filename[0],filename[1])

        self._gain_0 = blocks.multiply_const_cc (np.power (10,gain_dB/20))
        self._gain_1 = blocks.multiply_const_cc (np.power (10,gain_dB/20))

        self._chan_0 = channels.channel_model (frequency_offset=frequency_offset)
        self._chan_1 = channels.channel_model (frequency_offset=frequency_offset)

        # Load binary files
        self._src_0 = blocks.file_source (gr.sizeof_gr_complex,filename[0])
        self._src_1 = blocks.file_source (gr.sizeof_gr_complex,filename[1])

        if modulation_order==2:
            self._constellation = digital.constellation_bpsk ()
        else:
            # Setup constellation for data decoding part: QPSK for now per default
            self._constellation = digital.constellation_qpsk ()
        print ('Constellation:', self._constellation.points (), self._constellation.arity ())  # Debug

        self._receiver_0 = digital.constellation_decoder_cb (self._constellation.base ())
        self._receiver_1 = digital.constellation_decoder_cb (self._constellation.base ())

        # Frequency recovery
        if correct_frequency_offset:
            fll_ntaps = int(fll_ntaps_factor * K * samples_per_symbol)
            self.freq_recov = digital.fll_band_edge_cc (samples_per_symbol,alpha,fll_ntaps,2*np.pi/100.0)
            print ('Number of taps for FLL:', fll_ntaps)

        # Matched filter: with or without timing-offset correction
        if correct_timing_offset or correct_frequency_offset:
            nfilts = 32
            ntaps = nfilts * K * samples_per_symbol
            self._rx_rrc_taps = filter.firdes.root_raised_cosine (nfilts,nfilts*samples_per_symbol,1.0,alpha,ntaps)
            rrc_rx_filter = digital.pfb_clock_sync_ccf (samples_per_symbol,
                                                        2*np.pi/100.0, self._rx_rrc_taps,
                                                        nfilts, nfilts//2, 1.0)
        else:  # No timing offset
            ntaps = K * samples_per_symbol
            self._rx_rrc_taps = filter.firdes.root_raised_cosine (1,samples_per_symbol,1.0,alpha,ntaps)
            # The key for proper decimation is to set the rate to
            # samples_per_symbol
            # http://gnuradio.org/doc/doxygen/classgr_1_1filter_1_1kernel_1_1fft__filter__ccf.html
            rrc_rx_filter_0 = filter.fir_filter_ccf (samples_per_symbol,self._rx_rrc_taps)
            rrc_rx_filter_1 = filter.fir_filter_ccf (samples_per_symbol,self._rx_rrc_taps)

            # nfilts = 32
            # ntaps = nfilts * K * samples_per_symbol
            # self._rx_rrc_taps = filter.firdes.root_raised_cosine (nfilts,nfilts*samples_per_symbol,1.0,alpha,ntaps)
            # rrc_rx_filter = filter.pfb_arb_resampler_ccf (1/float (samples_per_symbol), self._rx_rrc_taps)

        print ('Number of taps for matched-filter:', ntaps)
        
        # Debug
        self._matched_filter_out_0 = blocks.vector_sink_c ()
        self._matched_filter_out_1 = blocks.vector_sink_c ()
        self._receiver_out_0 = blocks.vector_sink_b ()
        self._receiver_out_1 = blocks.vector_sink_b ()
        if correct_frequency_offset:
            self._f_frq = blocks.vector_sink_f ()
            self._f_phs = blocks.vector_sink_f ()
            self._f_err = blocks.vector_sink_f ()
        if correct_timing_offset or correct_frequency_offset:
            self._t_err = blocks.vector_sink_f ()
            self._t_rat = blocks.vector_sink_f ()
            self._t_phs = blocks.vector_sink_f ()

        self._r_sym_0 = blocks.vector_sink_c ()
        self._r_sym_1 = blocks.vector_sink_c ()

        # Connect
        core_flowgraph_0 = [self._src_0,self._gain_0,self._chan_0]
        core_flowgraph_1 = [self._src_1,self._gain_1,self._chan_1]
        if correct_frequency_offset:
            core_flowgraph_0.append ((self.freq_recov,0))
            core_flowgraph_0.append ((rrc_rx_filter,0))
            core_flowgraph_1.append ((self.freq_recov,1))
            core_flowgraph_1.append ((rrc_rx_filter,1))
        elif correct_timing_offset:
            core_flowgraph_0.append ((rrc_rx_filter,0))
            core_flowgraph_1.append ((rrc_rx_filter,1))
        else:
            core_flowgraph_0.extend ([rrc_rx_filter_0])
            core_flowgraph_1.extend ([rrc_rx_filter_1])

        core_flowgraph_0.extend ([self._receiver_0, blocks.null_sink (gr.sizeof_char)])
        core_flowgraph_1.extend ([self._receiver_1, blocks.null_sink (gr.sizeof_char)])
        self.connect (*core_flowgraph_0)
        self.connect (*core_flowgraph_1)
        if to_file is not None:
            if correct_frequency_offset or correct_timing_offset:
                self.connect ((rrc_rx_filter,0),blocks.file_sink (gr.sizeof_gr_complex,to_file[0]))
                self.connect ((rrc_rx_filter,1),blocks.file_sink (gr.sizeof_gr_complex,to_file[1]))
            else:
                self.connect (rrc_rx_filter_0,blocks.file_sink (gr.sizeof_gr_complex,to_file[0]))
                self.connect (rrc_rx_filter_1,blocks.file_sink (gr.sizeof_gr_complex,to_file[1]))
        # Connect debug
        self.connect (self._receiver_0,self._receiver_out_0)
        self.connect (self._receiver_1,self._receiver_out_1)
        if correct_frequency_offset:
            self.connect ((self.freq_recov,2), self._f_frq)
            self.connect ((self.freq_recov,3), self._f_phs)
            self.connect ((self.freq_recov,4), self._f_err)
        if correct_timing_offset or correct_frequency_offset:
            self.connect ((rrc_rx_filter,0), self._matched_filter_out_0)
            self.connect ((rrc_rx_filter,1), self._matched_filter_out_1)
            self.connect ((rrc_rx_filter,2), self._t_err)
            self.connect ((rrc_rx_filter,3), self._t_rat)
            self.connect ((rrc_rx_filter,4), self._t_phs)
        else:
            self.connect (rrc_rx_filter_0,self._matched_filter_out_0)
            self.connect (rrc_rx_filter_1,self._matched_filter_out_1)

if __name__ == '__main__':
    args = get_args ()

    assert len (args.filename) == 2

    tb = top_block (args.filename,
                    args.samples_per_symbol,args.rolloff_factor,args.K,
                    args.modulation_order,
                    args.correct_frequency_offset,args.correct_timing_offset,
                    args.fll_ntaps_factor,args.frequency_offset,
                    args.gain_dB,
                    seed=args.seed,to_file=args.to_file)
    tb.start ()
    tb.wait ()

    receiver_out_0 = np.array (tb._receiver_out_0.data ())
    receiver_out_1 = np.array (tb._receiver_out_1.data ())
    print ('Receiver 0 output length:',len (receiver_out_0))
    print ('Receiver 1 output length:',len (receiver_out_1))

    print ('Last 50 received symbols (0):',receiver_out_0[-50:])  # Debug
    print ('Last 50 received symbols (1):',receiver_out_1[-50:])  # Debug

    if not args.plot_offset_correction and not args.plot_mf_output:
        sys.exit (0)

    if args.correct_frequency_offset or args.correct_timing_offset:
        start = args.number_of_symbols-100
    else:
        start = 1
    mf_out_0 = np.array (tb._matched_filter_out_0.data ())[start:]
    mf_out_1 = np.array (tb._matched_filter_out_1.data ())[start:]
    print ('Matched filter output length (0):',len (mf_out_0),mf_out_0[:4])
    print ('Matched filter output length (1):',len (mf_out_1),mf_out_1[:4])

    if args.correct_frequency_offset:
        f = figure ()
        ax = f.add_subplot (211)
        ax.set_title ('Frequency recovery')
        frq = np.array(tb._f_frq.data()) / (2.0*np.pi)
        ax.plot (frq)
        ax = f.add_subplot (212)
        err = np.array(tb._f_err.data())
        ax.plot (err)
        tight_layout ()
        print ('Average frequency offset: {:.12f} (recommended: {:.12f})'.format (np.mean (frq),np.mean (frq)+args.frequency_offset))
        print ('Median frequency offset : {:.12f} (recommended: {:.12f})'.format (np.median (frq),np.median (frq)+args.frequency_offset))
        print ('95% frequency offset absolute error: {:.12f}'.format (np.percentile (np.abs (frq),0.95)))
    if args.correct_timing_offset or args.correct_frequency_offset:
        f = figure ()
        ax = f.add_subplot (311)
        ax.set_title ('Timing recovery')
        phs = np.array(tb._t_phs.data())
        ax.plot (phs)
        ax = f.add_subplot (312)
        err = np.array(tb._t_err.data())
        ax.plot (err)
        ax = f.add_subplot (313)
        rate = np.array(tb._t_rat.data())
        ax.plot (rate)
        tight_layout ()

    if args.plot_mf_output:
        f = figure ()
        ax = f.add_subplot (211)
        ax.set_title ('Matched filter output 0')
        ax.plot (mf_out_0.real,mf_out_0.imag,'.')
        ax.grid (True)
        ax = f.add_subplot (212)
        ax.plot (mf_out_0.real)
        ax.plot (mf_out_0.imag)
        ax.grid (True)
        tight_layout ()

        f = figure ()
        ax = f.add_subplot (211)
        ax.set_title ('Matched filter output 1')
        ax.plot (mf_out_1.real,mf_out_1.imag,'.')
        ax.grid (True)
        ax = f.add_subplot (212)
        ax.plot (mf_out_1.real)
        ax.plot (mf_out_1.imag)
        ax.grid (True)
        tight_layout ()

    show ()
