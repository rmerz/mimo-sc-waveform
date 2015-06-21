#!/usr/bin/env python
"""This script receives a pulse-shaped waveform and attempts to decode
it

"""
import argparse, sys
import numpy as np
from gnuradio import gr, digital, blocks, filter
from matplotlib.pyplot import figure, tight_layout, show

def get_args ():
    parser = argparse.ArgumentParser (description='Receive a pulse-shaped waveform.')
    parser.add_argument ('filename', type=str, nargs='+',
                         help='File to load for the waveform.')
    parser.add_argument ('-n','--number-of-symbols', type=int,
                         default=128, help='Number of "data" symbols expected per block.')
    parser.add_argument ('-s','--samples-per-symbol', type=int,
                         default=4, help='Number of samples per symbol (default is 4).')
    parser.add_argument ('-a','--rolloff-factor', type=float,
                         default=0.35, help='Excess bandwidth or roll-off factor (alpha, default is 0.35).')
    parser.add_argument ('-K', type=int,
                         default=12, help='Length of the pulse shaping filter in symbol units (default is 8).')
    parser.add_argument('-m', '--modulation-order',type=int,
                        default=4, help='Modulation order (default is 4 e.g. QPSK).')
    parser.add_argument ('--seed', type=int,
                         default=42, help='Seed of the random number generator used to generate data.')
    parser.add_argument ('-F','--to-file', type=str, nargs='+',
                         help='Save output of matched filter to binary file.')
    parser.add_argument ('-p','--periodicity', type=int,
                         help='Periodicity of the underlying symbol sequence.')
    parser.add_argument ('-f','--correct-frequency-offset', action='store_true',
                         help='Correct frequency offset.')
    parser.add_argument ('-c','--use-costas', action='store_true',
                         help='Finer frequency offset with Costas loop.')
    parser.add_argument ('--fll-ntaps-factor', type=float, default=1.5,
                         help='Multiplying factor for the number of taps for the FLL filter: factor * K * samples_per_symbol.')
    parser.add_argument ('-t','--correct-timing-offset', action='store_true',
                         help='Correct timing offset.')
    parser.add_argument ('--plot', action='store_true',
                         help='Display plots.')
    args = parser.parse_args ()
    return args

class top_block (gr.top_block):
    def __init__ (self,filename,
                  samples_per_symbol,alpha,K,
                  modulation_order=4,
                  correct_frequency_offset=False,correct_timing_offset=False,
                  fll_ntaps_factor=1.5,use_costas=False,
                  seed=42,to_file=None):
        gr.top_block.__init__(self)
        print 'Roll-off:', alpha

        # Load binary file
        self._src = blocks.file_source (gr.sizeof_gr_complex,filename[0])

        if modulation_order==2:
            self._constellation = digital.constellation_bpsk ()
        else:
            # Setup constellation for data decoding part: QPSK for now per default
            self._constellation = digital.constellation_qpsk ()
        print 'Constellation:', self._constellation.points (), self._constellation.arity ()  # Debug
        if use_costas:
            # Costas loop for finer correction of phase/frequency offset
            # (there can be residuals after the FLL)
            assert (modulation_order == 2 or modulation_order == 4)
            self._cl = digital.costas_loop_cc (2*np.pi/100.0, modulation_order)
        # Another option is to use digital.constellation_receiver_cb which contains the costas_loop as well
        self._receiver = digital.constellation_decoder_cb (self._constellation.base ())

        # Frequency recovery
        if correct_frequency_offset:
            fll_ntaps = int(fll_ntaps_factor * K * samples_per_symbol)
            self.freq_recov = digital.fll_band_edge_cc (samples_per_symbol,alpha,fll_ntaps,2*np.pi/100.0)
            print 'Number of taps for FLL:', fll_ntaps

        # Matched filter: with or without timing-offset correction
        if correct_timing_offset or correct_frequency_offset:
            nfilts = 32
            ntaps = nfilts * K * samples_per_symbol
            self._rx_rrc_taps = filter.firdes.root_raised_cosine (nfilts,nfilts*samples_per_symbol,1.0,alpha,ntaps)
            rrc_rx_filter = digital.pfb_clock_sync_ccf (samples_per_symbol,
                                                        2*np.pi/100.0, self._rx_rrc_taps,
                                                        nfilts, nfilts//2, 1.0)

        else:
            ntaps = K * samples_per_symbol
            self._rx_rrc_taps = filter.firdes.root_raised_cosine (1,samples_per_symbol,1.0,alpha,ntaps)
            # The key for proper decimation is to set the rate to
            # samples_per_symbol
            # http://gnuradio.org/doc/doxygen/classgr_1_1filter_1_1kernel_1_1fft__filter__ccf.html
            rrc_rx_filter = filter.fir_filter_ccf (samples_per_symbol,self._rx_rrc_taps)

            # nfilts = 32
            # ntaps = nfilts * K * samples_per_symbol
            # self._rx_rrc_taps = filter.firdes.root_raised_cosine (nfilts,nfilts*samples_per_symbol,1.0,alpha,ntaps)
            # rrc_rx_filter = filter.pfb_arb_resampler_ccf (1/float (samples_per_symbol), self._rx_rrc_taps)

        print 'Number of taps for matched-filter:', ntaps
        
        # Debug
        self._matched_filter_out = blocks.vector_sink_c ()
        self._receiver_out = blocks.vector_sink_b ()
        if correct_frequency_offset:
            self._f_frq = blocks.vector_sink_f ()
            self._f_phs = blocks.vector_sink_f ()
            self._f_err = blocks.vector_sink_f ()
        if correct_timing_offset or correct_frequency_offset:
            self._t_err = blocks.vector_sink_f ()
            self._t_rat = blocks.vector_sink_f ()
            self._t_phs = blocks.vector_sink_f ()

        if use_costas:
            self._cl_sym = blocks.vector_sink_c ()
            self._cl_frq = blocks.vector_sink_f ()
        self._r_sym = blocks.vector_sink_c ()

        # Connect
        core_flowgraph = [self._src]
        if correct_frequency_offset:
            core_flowgraph.append ((self.freq_recov,0))
        core_flowgraph.extend ([(rrc_rx_filter,0)])
        if use_costas:
            core_flowgraph.extend ([self._cl])
        core_flowgraph.extend ([self._receiver, blocks.null_sink (gr.sizeof_char)])
        self.connect (*core_flowgraph)
        if to_file is not None:
            if use_costas:
                self.connect (self._cl,blocks.file_sink (gr.sizeof_gr_complex,to_file[0]))
            else:
                self.connect ((rrc_rx_filter,0),blocks.file_sink (gr.sizeof_gr_complex,to_file[0]))
        # Connect debug
        self.connect ((rrc_rx_filter,0),self._matched_filter_out)
        self.connect (self._receiver,self._receiver_out)
        if correct_frequency_offset:
            self.connect (self._src,(self.freq_recov,1))
            self.connect ((self.freq_recov,1), (rrc_rx_filter,1))
            self.connect ((self.freq_recov,2), self._f_frq)
            self.connect ((self.freq_recov,3), self._f_phs)
            self.connect ((self.freq_recov,4), self._f_err)
        if correct_timing_offset or correct_frequency_offset:
            self.connect ((rrc_rx_filter,1), blocks.null_sink (gr.sizeof_gr_complex))
            self.connect ((rrc_rx_filter,2), self._t_err)
            self.connect ((rrc_rx_filter,3), self._t_rat)
            self.connect ((rrc_rx_filter,4), self._t_phs)

        if use_costas:
            self.connect (self._cl, self._cl_sym)
            self.connect ((self._cl,1), self._cl_frq)

if __name__ == '__main__':
    args = get_args ()

    tb = top_block (args.filename,
                    args.samples_per_symbol,args.rolloff_factor,args.K,
                    args.modulation_order,
                    args.correct_frequency_offset,args.correct_timing_offset,
                    args.fll_ntaps_factor,args.use_costas,
                    seed=args.seed,to_file=args.to_file)
    tb.start ()
    tb.wait ()

    receiver_out = np.array (tb._receiver_out.data ())
    print 'Receiver output length:',len (receiver_out)
    # # Is valid only if no frequency offset or timing offset had been applied
    # if not (args.correct_frequency_offset or args.correct_timing_offset):
    #     print '[DEPRECATED] Check received data assuming seed',args.seed
    #     np.random.seed (args.seed)
    #     offset = args.K*args.samples_per_symbol/2
    #     print '[DEPRECATED] Offset:', offset
    #     replica = np.random.randint (0,tb._constellation.arity (),args.number_of_symbols)[-(len (receiver_out)-offset):]
    #     # print replica,receiver_out[offset:]  # Debug
    #     difference = replica-receiver_out[offset:]
    #     print '[DEPRECATED] Difference:',difference
    #     print '[DEPRECATED] Sum of difference:',np.sum (np.abs (np.sign (difference)))
    print 'Last 50 received symbols:',receiver_out[-50:]  # Debug
    if args.periodicity is not None:
        # The transmitted signal has a symbol period of args.periodicity
        periodicity_check = receiver_out[-args.periodicity:]-receiver_out[-2*args.periodicity:-args.periodicity]
        print 'Periodicity check:', periodicity_check, np.sum (np.abs (np.sign (periodicity_check)))

    if not args.plot:
        sys.exit (0)

    if args.correct_frequency_offset or args.correct_timing_offset:
        start = args.number_of_symbols-100
    else:
        start = 1
    mf_out = np.array (tb._matched_filter_out.data ())[start:]
    print 'Matched filter output length:',len (mf_out)

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

    f = figure ()
    ax = f.add_subplot (211)
    ax.set_title ('Matched filter output')
    ax.plot (mf_out.real,mf_out.imag,'.')
    ax.grid (True)
    ax = f.add_subplot (212)
    ax.plot (mf_out.real)
    ax.plot (mf_out.imag)
    ax.grid (True)
    tight_layout ()

    if args.use_costas:
        f = figure ()
        ax = f.add_subplot (211)
        ax.set_title ('Finer correction (Costas loop)')
        frq = np.array (tb._cl_frq.data()) / (2.0*np.pi)
        ax.plot (frq)
        ax = f.add_subplot (212)
        data_sym = np.array (tb._cl_sym.data())
        ax.plot (data_sym.real, data_sym.imag, "rx")
        if len (data_sym) > 1000:
            ax.plot (data_sym.real[-1000:], data_sym.imag[-1000:], "bo")
        tight_layout ()

    show ()
