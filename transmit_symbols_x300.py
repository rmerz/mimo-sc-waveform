#!/usr/bin/env python
"""This script loads a symbol sequence from binary file, performs
pulse-shaping and transmit its via a USRP x300.

"""
from __future__ import print_function
import time, sys, argparse
from gnuradio import gr, analog, blocks, filter, uhd
import numpy as np


def get_args ():
    parser = argparse.ArgumentParser(description='Using ')
    parser.add_argument('-f','--filename', type=str, nargs='+',
                        help='File(s) to load for the waveform generation.')
    parser.add_argument('--ip-address', default='192.168.13.2', type=str,
                        help='IP address of the USRP (default is 192.168.13.2).')
    parser.add_argument('--center-frequency', default=2600.0e6, type=float,
                        help='Center frequency in Hz (default is 2600.0 MHz).')
    parser.add_argument('--sampling-rate', default=0.5e6, type=float,
                        help='Sampling rate of the signal in S/s (default is 0.5 MS/s).')
    parser.add_argument('--clock-rate', default=200e6, type=float,
                        help='Clock-rate of the underlying hardware (default is 200 MS/s, supported is also 184.32e6).')
    parser.add_argument('--pa-gain', default=0, type=float,
                        help='Gain of the PA on the frontends (default is 0 dB, range is 0 to 20 dB).\tDisable safety to use it.')
    parser.add_argument('--disable-safety', action='store_true',
                        help='Enables to increase the PA gain.')
    parser.add_argument('--no-dsp-offset', action='store_true',
                        help='Deactivate default DSP offset equal to the sample-rate.')
    parser.add_argument('--sine-frequency', type=float,
                        help='Instead of file-based inputs, activates a sine signal with given frequency.')
    parser.add_argument('--scaling', nargs='+', default=[1.0], type=float,
                        help='Scaling of the complex signals (default is 1.0 for each channel).')
    parser.add_argument('-n','--number-of-channels', default=2, type=int,
                        help='Number of TX channels enabled (default is 2).')
    parser.add_argument('--internal-clock', action='store_true',
                        help='Disable the attempt to lock to an external clock source.')
    parser.add_argument('-s','--samples-per-symbol', type=int,
                        default=4, help='Number of samples per symbol (default is 4).')
    parser.add_argument('-a','--rolloff-factor', type=float,
                        default=0.35, help='Excess bandwidth or roll-off factor (alpha, default is 0.35).')
    parser.add_argument('-K', type=int,
                        default=12, help='Length of the pulse shaping filter in symbol units (default is 12).')
    args = parser.parse_args()
    print (args)  # Debug
    return args

class my_top_block (gr.top_block):
    def __init__ (self):
        args = get_args ()
        gr.top_block.__init__ (self)

        self._usrp = None
        self._channels_id_list = None
        self._m_board = 0

        # Init and configure USRP
        self.init_usrp (self._m_board,args.ip_address,args.clock_rate,args.sampling_rate,args.number_of_channels,args.internal_clock)
        # Configure front-ends
        if args.pa_gain > 0 and args.disable_safety:
            self.configure_front_ends (args.pa_gain)
        else:
            print ('Safety engaged, will not increase the PA gain.')
            self.configure_front_ends (0)
        # Tune
        self.tune_front_ends (args.center_frequency,args.no_dsp_offset,self._m_board)

        # Load files or activate sine-wave
        if args.sine_frequency is None:
            ntaps = args.K * args.samples_per_symbol
            self._tx_rrc_taps = filter.firdes.root_raised_cosine (args.samples_per_symbol, args.samples_per_symbol, 1.0,args.rolloff_factor, ntaps)
            print ('Number of taps for pulse-shaping: {:d} ({:d} sps, alpha {:f})'.format (ntaps,args.samples_per_symbol,args.rolloff_factor))
            # Load binary file
            for k in np.arange (len (args.filename)):
                print ('Load file {:s} on channel {:d}'.format (args.filename[k],self._channels_id_list[k]))
                print ('Scaling:',args.scaling)
                self.connect (blocks.file_source (gr.sizeof_gr_complex,args.filename[k],repeat=True),
                              blocks.multiply_const_cc (args.scaling[k]),
                              filter.interp_fir_filter_ccf (args.samples_per_symbol,self._tx_rrc_taps),
                              (self._usrp,self._channels_id_list[k]))
        else:
            self._signal_source = analog.sig_source_c (args.sampling_rate, analog.GR_SIN_WAVE, args.sine_frequency, args.scaling[0])
            # Connect all blocks and run
            for channel_id in self._channels_id_list:
                self.connect (self._signal_source,(self._usrp,channel_id))
        print ('Transmission setup completed.')

    def init_usrp (self,mboard,ip_address,clock_rate,sampling_rate,number_of_channels,internal_clock):
        """Setup the USRP"""
        self._channels_id_list = range (number_of_channels)
        # Init USRP: See http://gnuradio.org/doc/doxygen/page_uhd.html
        self._usrp = uhd.usrp_sink (device_addr='addr={:s},master_clock_rate={:f}'.format (ip_address,clock_rate),
                                    stream_args=uhd.stream_args (cpu_format="fc32",
                                                                 otw_format="sc16",
                                                                 channels=self._channels_id_list,))

        if not internal_clock:
            # Try to set an external clock source if available
            try:
                print ('Try external clock.')
                self._usrp.set_clock_source('external',mboard=mboard)
            except RuntimeError as e:
                print (e)
                print ('Could not lock to external clock source. Lock back to internal clock source.')
                self._usrp.set_clock_source('internal',mboard=mboard)

        while not self._usrp.get_mboard_sensor ("ref_locked", mboard).to_bool ():
            print ('Wait for ref_locked')
            time.sleep (1)

        self._usrp.set_time_unknown_pps (uhd.time_spec_t (0.0))
        time.sleep (2.0)  # Wait for clocks to settle
        print ('Time setup finished: {:f}, {:f}'.format (self._usrp.get_time_now(mboard=mboard).get_real_secs (),self._usrp.get_time_last_pps (mboard=mboard).get_real_secs ()))

        self._usrp.set_samp_rate (sampling_rate)
        self._usrp.set_subdev_spec ('A:0 B:0',mboard=mboard)
        print ('USRP initialized:')
        print ('- Configured clock rate: {:f}'.format (self._usrp.get_clock_rate ()))
        print ('- Configured clock source: {:s}'.format (self._usrp.get_clock_source (mboard=mboard)))
        print ('- Configured sampling rate: {:f}'.format (self._usrp.get_samp_rate ()))

    def configure_front_ends (self,pa_gain):
        """Setup the antenna port, bandwidth and PA gain"""
        assert (self._usrp is not None and self._channels_id_list is not None)
        for channel_id in self._channels_id_list:
            # Set other parameters
            self._usrp.set_antenna ('TX/RX',chan=channel_id)
            self._usrp.set_bandwidth (1e6,channel_id)  # Set bandwidth to 1 MHz
            self._usrp.set_gain (pa_gain,channel_id)
            # Debug
            print (self._usrp.get_bandwidth_range (channel_id))
            # print ('Full bandwidth range of the front-end: {:f} MHz'.format (float (self._usrp.get_bandwidth_range (channel_id)[0])/1e6))
            print ('Configured bandwidth of the front-end: {:f} Hz'.format (self._usrp.get_bandwidth (channel_id)))
            # while not self._usrp.get_sensor ("lo_locked", channel_id).to_bool ():
            #     print ('Wait for lo_locked')
            #     time.sleep (1)

    def tune_front_ends (self,center_frequency,no_dsp_offset=False,mboard=0):
        """Tune the front-ends"""
        assert (self._usrp is not None and self._channels_id_list is not None)

        # When was last PPS observed?
        last_pps_time_seconds = self._usrp.get_time_last_pps (mboard).get_real_secs ()
        print ('Pre-tune: last PPS {:f}'.format (last_pps_time_seconds))
        future_cmd_time = uhd.time_spec_t (last_pps_time_seconds + 2.0)

        # Make sure the next commands are executed at future_cmd_time
        self._usrp.set_command_time (future_cmd_time,mboard)

        for channel_id in self._channels_id_list:
            # Can simply be: self._usrp.set_center_freq (2655e6)
            if no_dsp_offset:
                dsp_offset = 0
            else:
                dsp_offset = self._usrp.get_samp_rate ()  # To get any leakage from the receive LO out of the way
                print ('DSP offset:',dsp_offset)
            tune_result = self._usrp.set_center_freq (uhd.tune_request (target_freq=center_frequency,
                                                                        rf_freq_policy=uhd.tune_request.POLICY_MANUAL,
                                                                        rf_freq=center_frequency-dsp_offset,
                                                                        dsp_freq_policy=uhd.tune_request.POLICY_AUTO),channel_id)
            if tune_result is None:
                print ('Could not set center frequency.')
                sys.exit (-1)
        # We're done
        self._usrp.clear_command_time (mboard)
        print ('Post-tune: last PPS {:f}'.format (self._usrp.get_time_last_pps (mboard).get_real_secs ()))

        time.sleep (3.0) # Wait for commands to have executed
        for channel_id in self._channels_id_list:
            print ('Front-end/channel {:d}: Setting center frequency was successful.'.format (channel_id))
            print (tune_result.actual_rf_freq, tune_result.actual_dsp_freq)
            print ('Frequency set to: {:f}'.format (self._usrp.get_center_freq (chan=channel_id)))



if __name__ == '__main__':
    try:
        my_top_block().run()
    except [[KeyboardInterrupt]]:
        pass
