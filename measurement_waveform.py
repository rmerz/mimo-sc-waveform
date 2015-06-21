"""Module that contains classes and helper to build a measurement
waveform and process it.

"""
from __future__ import print_function
import sys
import numpy as np
from scipy.linalg import hadamard


def lte_gold_sequence_generator (M_PN,c_init):
    """Generation LTE length-31 Gold sequence

    M_PN  : sequence length
    c_init: initialization constant

    See section 7.2 of
    http://www.etsi.org/deliver/etsi_ts/136200_136299/136211/12.03.00_60/ts_136211v120300p.pdf
    (section 6.10.1.1)

    """
    N_c = 1600
    # print (c_init)  # Debug
    x_1 = np.zeros (32,dtype=int)  # 31 + 1
    x_1[0] = 1
    x_2 = np.zeros (32,dtype=int)  # 31 + 1
    P = 2**np.arange (0,31)
    for k in np.arange (30,-1,-1):
        if 2**k + np.sum (x_2[:-1] * P) <= c_init:
            x_2[k] = 1
    for k in np.arange (1600):
        x_1[31] = x_1[3] ^ x_1[0]
        x_2[31] = x_2[3] ^ x_2[2] ^ x_2[1] ^ x_2[0]
        x_1 = np.roll (x_1,-1)
        x_2 = np.roll (x_2,-1)
    c = -np.ones (M_PN)
    for k in np.arange (M_PN):
        x_1[31] = x_1[3] ^ x_1[0]
        x_2[31] = x_2[3] ^ x_2[2] ^ x_2[1] ^ x_2[0]
        c[k] = x_1[0] ^x_2[0]
        x_1 = np.roll (x_1,-1)
        x_2 = np.roll (x_2,-1)
    return c

def _3gpp_lte_crs_generator (N_RB=100,N_cell_ID=0,n_s=0,l=0,N_CP=1):
    """Generate 3GPP LTE cell-specific reference signal. Returns a sequence of length 2*N_RB

    N_RB     : number of resource blocks (can be abused as length
    N_cell_ID: cell ID
    n_s      : slot number within a radio frame (0 to 19)
    l        : OFDM symbol number within a slot (0 to 4 typically)
    N_CP : whether a normal or extended cyclic prefix is used (assume
           normal by default, e.g. N_CP=1)

    See
    http://www.etsi.org/deliver/etsi_ts/136200_136299/136211/12.03.00_60/ts_136211v120300p.pdf
    (section 6.10.1.1)

    """
    sequence_length = 2*(2*N_RB)+1
    c_init = 1024*(7*(n_s+1)+l+1)*(2*N_cell_ID+1)+(2*N_cell_ID)+N_CP
    c = lte_gold_sequence_generator (sequence_length,c_init)
    # print (c)  #  Debug
    crs_sequence = np.zeros (2*N_RB,dtype=np.complex)
    for m in np.arange (2*N_RB):
        crs_sequence[m] = ((1-2*c[2*m]) + 1j*(1-2*c[2*m+1]))/np.sqrt (2)

    return crs_sequence

class measurementSymbols:
    def __init__ (self,payload_length=1024,
                  mimo_streams=1,
                  periodicity=None,
                  repeat_symbols=None,
                  preamble_seq_length=32,preamble_root_length=64,
                  pilot_seq_length=512,pilot_root_length=4,
                  seed=42):
        """Generate symbols to be used as a measurement waveform

        payload_length   : number of data symbols to generate
        mimo_streams     : number of streams to generate (default to 1)
        periodicity      : [DEPRECATED] if set to an integer, the data symbols will be a
                           periodic sequence with period periodicity
                           (kept for compatibility with previous version)
        repeat_symbols   : symbols consist of a preamble and a payload. If
                           set to an integer, repeats the symbols repeat_symbols
                           times.
        preamble_seq_length : number of repetitions of the preamble root (default to 32)
        preamble_root_length: length of the preamble root (default to 64)
        pilot_seq_length : number of repetitions of the pilot root (default to 32)
        pilot_root_length: length of the pilot root (default to 64)
        seed             : sets the seed of the numpy random number generator

        QPSK is assumed to be the underlying constellation for now.

        """
        self._seed = seed
        print ('[Debug] RNG seed:', self._seed)
        np.random.seed (self._seed)
        self._payload_length = payload_length

        self._mimo_streams = mimo_streams

        self._preamble_seq_length = preamble_seq_length
        self._preamble_root_length = preamble_root_length
        # +1 because of start frame delimiter
        print ('[Debug] Length of preamble:', (self._preamble_seq_length+1)*self._preamble_root_length)
        self._pilot_seq_length = pilot_seq_length
        self._pilot_root_length = pilot_root_length
        print ('[Debug] Length of pilot:', (self.pilot_length))

        self._periodicity = periodicity
        self._repeat_symbols = repeat_symbols

        self._preamble_root = None
        self._preamble = None
        self._sfd = None
        self._pilot_root = None
        self._pilot = None
        self._pilot_matrix_root = None
        self._pilot_matrix = None
        self._data = None
        self._payload = None
        self._symbols = None  # Place-holder for the numpy array containing the symbols

        self._arity = 4  # Assumes QPSK only for now
        # np.complex64 is used because this is used by gnuradio file_sink
        self._qpsk_mapping = np.array ([-1-1j,1-1j,-1+1j,1+1j],dtype=np.complex64) / np.sqrt (2)
        self._qpsk_demapping = np.array ([[0,2],[1,3]],dtype=np.int)
        assert (self._arity == len (self._qpsk_mapping))

    @property
    def data (self):
        return self._data

    @property
    def payload (self):
        return self._payload

    @property
    def preamble (self):
        return self._preamble

    @property
    def pilot (self):
        return self._pilot

    @property
    def pilot_length (self):
        return self._pilot_seq_length*self._pilot_root_length

    @property
    def symbols (self):
        return self._symbols

    def generate_preamble (self,preamble_id=43,sfd_id=97,
                           single_preamble=False):
        """Generate a preamble. Comprises both a preamble sequence and a start
        frame delimiter (SFD) sequence

        preamble_id: identifier of the preamble root
        sfd_id     : identifier of the SFD root
        single_preamble: preamble is generated only for the first stream

        The preamble is the same for each MIMO stream

        """
        self._preamble_root = np.tile (_3gpp_lte_crs_generator (self._preamble_root_length/2,preamble_id),[self._mimo_streams,1])
        if self._mimo_streams > 1 and single_preamble:
            print ('Single preamble mode')
            self._preamble_root[1:,:] = 0
        self._preamble = np.tile (self._preamble_root, self._preamble_seq_length)
        self._sfd = np.tile (_3gpp_lte_crs_generator (self._preamble_root_length/2,sfd_id),[self._mimo_streams,1])
        if self._mimo_streams > 1 and single_preamble:
            self._sfd[1:,:] = 0

    def generate_pilot_matrix (self,pilot_id=None):
        """Generate matrices of pilot sequences (root and full) to enable MIMO channel estimation

        """
        if pilot_id is None:
            pilot_id = [-1,-2]
        assert len (pilot_id) > 1
        if self._pilot_matrix is None and self._pilot_matrix_root is None:
            self._pilot_matrix_root = np.zeros ((len (pilot_id),self._pilot_root_length),dtype=np.complex)
            for k in np.arange (len (pilot_id)):
                self.generate_pilot (pilot_id[k])
                self._pilot_matrix_root[k,:] = self._pilot_root
            self._pilot_matrix = np.tile (self._pilot_matrix_root,self._pilot_seq_length)


    def generate_pilot (self,pilot_id=-1):
        """Generate a pilot sequence to enable channel estimation

        Do self.generate_pilot (0) and s0 = self._pilot followed by
        self.generate_pilot (1) and s1 = self._pilot. Then compare
        np.dot (s0,np.conjugate (s0)) with np.dot (s0,np.conjugate
        (s1))

        The pilot_id is decremented for each stream

        """
        assert (pilot_id <= 2*self._pilot_root_length)
        pilot_id_list = np.cumsum (np.ones (self._mimo_streams,dtype=int)*pilot_id)
        pilot_base = np.array (hadamard (2*self._pilot_root_length)[pilot_id_list,:])
        assert (pilot_base.shape[1] == 2*self._pilot_root_length)
        self._pilot_root = (pilot_base[:,:self._pilot_root_length] + 1j*pilot_base[:,self._pilot_root_length:])/np.sqrt (2)
        print ('[Debug] pilot list ({:d}/{:d}):'.format (self._pilot_root_length,self._pilot_root.shape[1]),pilot_id_list)
        self._pilot = np.tile (self._pilot_root, self._pilot_seq_length)

    def generate_symbols (self,with_preamble=True,with_pilot=True,single_preamble=False,debug=False):
        """Generate measurement symbols. They comprise both preamble and data
        symbols

        with_preamble: do add the preamble if already configured (default is True)
        with_pilot   : do add the pilot if already configured (default is True)
        single_preamble: the preamble is sent on the first stream only (0 on the other streams)
        debug:       : returns the generated data sequence

        Generation of data-symbols is similar to gnuradio.

        """
        if self._periodicity is not None:
            self._data = np.tile (np.random.randint (0,self._arity,size=self._periodicity) % self._arity,[self._mimo_streams,self._payload_length//self._periodicity])
        else:
            self._data = np.tile (np.random.randint (0,self._arity,size=self._payload_length),[self._mimo_streams,1])
        # To QPSK symbols for the data part
        self._payload = self._qpsk_mapping [self._data]

        self._symbols = self._payload
        # Add the pilot if need be
        if with_pilot:
            self.generate_pilot ()
            self._symbols = np.concatenate ([self._pilot,self._symbols],axis=1)
        # Add the preamble if need be
        if with_preamble:
            self.generate_preamble (single_preamble=single_preamble)
            self._symbols = np.concatenate ([self._preamble,self._sfd,self._symbols],axis=1)

        # Repeat the symbols if need be
        if self._repeat_symbols is not None:
            self._symbols = np.tile (self._symbols, self._repeat_symbols)

        if debug:
            return self._data

    def preamble_synchronization (self,received_symbols,gain=1,check_needed=3):
        """Use the preamble to recover symbol timing, e.g. when the payload
        begins. Returns the offset to access the payload.

        received_symbols: a numpy array containing the received symbols
        gain:           : number of repetition of the preamble root sequence to form a synchronization mask
        check_needed    : number of consecutive times where synchronization with the sync. mask must match

        This is a prerequisite for channel estimation

        """
        self.generate_preamble ()
        sync_mask = np.tile (self._preamble_root,gain)
        print ('[Debug] gain {:d} x {:d}, check {:d}'.format (gain,self._preamble_root.shape[1],check_needed))
        # print (sync_mask.shape)
        offset = 0
        id_max = None
        id_max_prev = -1
        max_count = 0
        while offset <= len (received_symbols) and offset+sync_mask.shape[1] <= len (received_symbols):
            # Non-coherent: we need the same mode to detect offsets
            rx_conv = np.abs (np.correlate (received_symbols[offset:offset+sync_mask.shape[1]]**2,sync_mask[0]**2,mode='same'))  # valid
            # Check where the maximum is
            id_max = np.argmax (rx_conv)
            if rx_conv[id_max] > 0 and id_max_prev == id_max:  # FIXME: need to add a condition for amplitude of correlation
                # Seems we found something
                max_count += 1
                if max_count == check_needed:
                    # We're pretty sure this is a preamble
                    break
            else:
                max_count = 0
            id_max_prev = id_max
            print ('[Debug] offset: {:d} id_max {:d} ({:d}) checked {:d}'.format (offset,id_max,len (rx_conv),max_count))
            offset += sync_mask.shape[1]
        if max_count == 0:  # Remainder of received_symbols was not long enough
            return None
        # We should be done with the sync. (but will verify one last time)
        print ('[Debug] Identified argmax: {:d} (checked {:d}), perform last check'.format (id_max,max_count))
        # Verify one last time by moving at the beginning of the preamble
        offset = offset + id_max - sync_mask.shape[1]/2
        rx_conv = np.abs (np.correlate (received_symbols[offset:offset+sync_mask.shape[1]]**2,sync_mask[0]**2,mode='same'))
        if np.argmax (rx_conv) == sync_mask.shape[1]/2:
            print ('Sync. confirmed, offset: ({:d})'.format (offset))
        else:
            print ('Sync. verification failed')
            return None
        # Now looking for SFD
        sfd_check_count = 0
        sfd_found = False
        while offset <= len (received_symbols) and sfd_check_count <= self._pilot_seq_length:
            rx_conv = np.abs (np.correlate (received_symbols[offset:offset+self._sfd.shape[1]]**2,self._sfd[0]**2,mode='same'))
            # Check where the maximum is
            id_max = np.argmax (rx_conv)
            if id_max == self._sfd.shape[1]/2:
                print ('SFD found')
                sfd_found = True
                break
            sfd_check_count += 1
            offset += self._sfd.shape[1]
        if sfd_found:
            # Jump to beginning of channel estimation
            offset += self._sfd.shape[1]
            return offset
        else:
            return None

    def mimo_channel_estimation (self,received_symbols,pilot_id=None):
        """Perform LS channel estimation

        """
        self.generate_pilot_matrix (pilot_id)
        # Over the full pilot length
        P_inv = np.linalg.pinv (self._pilot_matrix)
        H_pilot = np.dot (received_symbols,P_inv)
        print ('[Debug] H_pilot and angle (H_pilot)')
        print (H_pilot)
        print (np.angle (H_pilot))
        # For each pilot root sequence
        H_root = np.zeros ((len (pilot_id),received_symbols.shape[0],self._pilot_seq_length),dtype=np.complex)
        offset = 0
        P_inv = np.linalg.pinv (self._pilot_matrix_root)
        for k in np.arange (self._pilot_seq_length):
            H_root[:,:,k] = np.dot (received_symbols[:,offset:offset+self._pilot_root_length],P_inv)
            offset += self._pilot_root_length
        mean_foffset_per_root = np.mean (np.angle (H_root),axis=2)
        mean_fdrift_per_root = np.mean (np.diff (np.angle (H_root),axis=2))
        return mean_foffset_per_root,mean_fdrift_per_root,H_pilot,H_root

    def rotate_pilot (self,received_symbols,H_root):
        # Construct a vector of rotation to apply to received symbols
        rotation = np.zeros (received_symbols.shape,dtype=np.complex)
        symbol_offset = 0
        # Iterate over all channel matrices
        for k in np.arange (H_root.shape[2]):
            phase_offset = np.diag (np.angle (H_root[:,:,k]))
            rotation[0,symbol_offset:symbol_offset+self._pilot_root_length] = np.exp (-1j*phase_offset[0])
            rotation[1,symbol_offset:symbol_offset+self._pilot_root_length] = np.exp (-1j*phase_offset[1])
            symbol_offset += self._pilot_root_length
        return received_symbols*rotation

    def print_condition_number (self,H_root):
        condition_number_dB = list ()
        for k in np.arange (H_root.shape[2]):
            condition_number_dB.append (20*np.log10 (np.linalg.cond (H_root[:,:,k])))
        print ('CN dB:',np.median (condition_number_dB))

    def siso_channel_estimation (self,received_symbols,pilot_id=-1):
        """Perform channel estimation

        received_symbols: the sequence of received symbols where the channel estimation pilot can be found

        Note that np.dot (s0,np.conjugate (s0)) is equivalent to
        np.correlate (s0,s0)

        """
        self.generate_pilot (pilot_id)
        # First calculate on each pilot root
        h = np.zeros (self._pilot_seq_length,dtype=np.complex)
        offset = 0
        # First compute an average over each pilot root
        for k in np.arange (self._pilot_seq_length):
            h[k] = np.dot (received_symbols[offset:offset+self._pilot_root_length],np.conjugate (self._pilot_root[0]))
            offset += self._pilot_root_length

        median_amplitude = np.median (h.real)
        print ('[Debug] median amplitude:',median_amplitude)
        median_fdrift_per_symbol = np.median (np.diff (np.angle (h)))/self._pilot_root_length
        print ('[Debug] median phase drift per symbol:',median_fdrift_per_symbol)
        median_foffset_per_symbol = np.median (np.angle (h))
        print ('[Debug] median phase offset per symbol:',median_foffset_per_symbol)

        # # Second compute an average over the full pilot
        # h_avg = np.dot (received_symbols,np.conjugate (self.pilot))
        # print ('[Debug] avg. angle: {:.6f} {:.6f}'.format (np.angle (h_avg),np.mean (np.angle (h))))
        return median_foffset_per_symbol,median_fdrift_per_symbol,h

    def correct_phase_offset (self,received_symbols,phase_offset,phase_drift_per_symbol):
        last_phase_offset = phase_offset+phase_drift_per_symbol*len (received_symbols)
        return received_symbols*np.exp (-1j*phase_offset)*np.exp (-1j*phase_drift_per_symbol*np.cumsum (np.ones (len (received_symbols)))),last_phase_offset

    def decode_symbols (self,received_symbols):
        """Recover symbols from the output of the matched-filter"""
        self._data = self._qpsk_demapping[((1.0+np.sign (received_symbols.real))/2).astype (int),((1.0+np.sign (received_symbols.imag))/2).astype (int)]

    def check_periodicity (self):
        """Check periodicity property using the last two periods. Assumes that
        the periodicity option has been used.

        """
        assert (self._data.shape[1] > 2*self._periodicity)
        # rtol and atol can be set to zero because we have integers
        return np.allclose (self._data[-self._periodicity:],self._data[-2*self._periodicity:-self._periodicity],rtol=0,atol=0)

    def to_file (self,filename_template='measurement_symbols'):
        """Save sequence of symbols to file in binary format

        The format should be compatible with gnuradio

        """
        for k in np.arange (self._mimo_streams):
            if self._symbols is not None:
                self._symbols[k].astype ('complex64').tofile (filename_template+'_{:d}.bin'.format (k))
            else:
                print ('No valid symbols found, abort to_file operation.')

class symbolReceiver ():
    def __init__ (self,payload_length=1024,periodicity=None,
                  pilot_seq_length=512,pilot_root_length=4,
                  seed=42):
        self._payload_length = payload_length
        self._periodicity= periodicity
        self._pilot_seq_length = pilot_seq_length
        self._pilot_root_length = pilot_root_length
        self._seed = seed

        self._rx_symbols = None

        self._mw = measurementSymbols (payload_length=self._payload_length,periodicity=self._periodicity,
                                       pilot_seq_length=self._pilot_seq_length,pilot_root_length=self._pilot_root_length,
                                       seed=self._seed)

    @property
    def rx_symbols (self):
        return self._rx_symbols

    @property
    def pilot_seq_length (self):
        return self._pilot_seq_length

    def load_stream (self,filename):
        """Load a single stream of data generated by gnuradio.

        filename: file to load

        Care needs to be taken with respect to the data format

        """
        # gnuradio saves data in np.complex64 format
        self._rx_symbols = np.fromfile (filename,dtype=np.complex64,count=-1)
        print ('Received {:d} symbols'.format (len (self._rx_symbols)))

    def load_multiple_streams (self,filename):
        """Load N streams of data generated jointly by gnuradio.

        filename: array of files to load

        Care needs to be taken with respect to the data format

        """
        self._rx_symbols = list ()
        for k in np.arange (len (filename)):
            self._rx_symbols.append (np.fromfile (filename[k],dtype=np.complex64,count=-1))
        self._rx_symbols = np.array (self._rx_symbols)
        print ('Received {:d} symbols ({:d}x{:d})'.format (self._rx_symbols.shape[1],self._rx_symbols.shape[0],self._rx_symbols.shape[1]))

    def mimo_2x2_sync_and_decode (self,gain=1,check_needed=3,correct_phase=False,pilot_id=None,save_channels=False,count=None):
        """Perform frame synchronization, channel estimation with phase correction and display data on two received streams assuming a 2x2 MIMO channel

        gain         : gain for frame synchronization
        check_needed : how many times must sync with mask be successful
        correct_phase: set to True to perform residual phase offset correction
        pilot_id: a list of pilot IDs can be specified. Channel
                  estimation for phase correction will be run using
                  the first pilot ID. Additional pilot ID can be
                  specified to obtain further channel estimation
                  information
        save_channels: return an array of channel matrices (default is False)

        """
        more_data = True
        offset = 0
        k = 0
        phase_offset = None
        last_phase_offset = None
        phase_drift = None
        if pilot_id is None:
            pilot_id = [-1,-2]
        assert len (pilot_id) == 2
        channels = list ()
        while more_data:
            if offset > self._rx_symbols.shape[1]:
                break
            # Run synchronization: on sum of received streams (potentially using quite a bit of memory)
            relative_offset = self._mw.preamble_synchronization (self._rx_symbols[0,offset:]+self._rx_symbols[1,offset:],gain=gain,check_needed=check_needed)
            if relative_offset is not None:
                next_offset=offset+relative_offset
            else:
                offset += self._mw._pilot_root_length
                continue
            if len (self._rx_symbols[0,next_offset:]) < self._mw.pilot_length:
                # No more symbols
                break
            print ('[Debug] payload starts at {:d} (pilot is {:d} symbols)'.format (next_offset,self._mw.pilot_length))
            # Channel estimation
            _,_,_,H_root = self._mw.mimo_channel_estimation (self._rx_symbols[:,next_offset:next_offset+self._mw.pilot_length],pilot_id)
            self._mw.print_condition_number (H_root)
            if save_channels:
                channels.append (H_root)
            if correct_phase:
                # Rotate pilot and verify
                rotated_pilot = self._mw.rotate_pilot (self._rx_symbols[:,next_offset:next_offset+self._mw.pilot_length],H_root)
                print ('Check rotation')
                _,_,_,H_root_rot = self._mw.mimo_channel_estimation (rotated_pilot,pilot_id)
                self._mw.print_condition_number (H_root_rot)
            next_offset += self._mw.pilot_length
            # Check whether there is still something to decode
            if len (self._rx_symbols[0,next_offset:]) > self._payload_length:
                # Decode symbols (suboptimal implementation for now)
                self._mw.decode_symbols (self._rx_symbols[:,next_offset:next_offset+self._payload_length])
                offset = next_offset + self._payload_length
            else:
                print ('Partial decoding: {:d} < {:d}'.format (len (self._rx_symbols[0,next_offset:]),self._payload_length))
                self._mw.decode_symbols (self._rx_symbols[:,next_offset:])
                more_data = False

            print ('[Debug] {:d}:'.format (k),self._mw.data[:,:10],self._mw.data[:,-10:])  # Debug
            if count is not None and k == count:
                break
            k += 1
        if save_channels:
            return channels
        else:
            return None

    def siso_sync_and_decode (self,gain=1,skip_pilot=False,correct_phase=False,pilot_id=None):
        """Perform frame synchronization, channel estimation with phase correction and display data on a single received stream

        gain         : gain for frame synchronization
        skip_pilot   : do not perform channel estimation
        correct_phase: set to True to perform residual phase offset correction
        pilot_id: a list of pilot IDs can be specified. Channel
                  estimation for phase correction will be run using
                  the first pilot ID. Additional pilot ID can be
                  specified to obtain further channel estimation
                  information

        """
        more_data = True
        offset = 0
        k = 0
        phase_offset = None
        last_phase_offset = None
        phase_drift = None
        if pilot_id is None:
            pilot_id = [-1]
        while more_data:
            # Run synchronization
            next_offset=offset+self._mw.preamble_synchronization (self._rx_symbols[offset:],gain=gain)
            if len (self._rx_symbols[next_offset:]) < self._mw.pilot_length:
                # No more symbols
                break
            # Correct residual phase rotation
            if correct_phase and phase_drift is not None:
                self._rx_symbols[offset:next_offset],_ = self._mw.correct_phase_offset (self._rx_symbols[offset:next_offset],last_phase_offset,phase_drift)
            print ('[Debug] payload starts at {:d}'.format (offset))
            # Channel estimation
            if not skip_pilot:
                phase_offset,phase_drift,_ = self._mw.siso_channel_estimation (self._rx_symbols[next_offset:next_offset+self._mw.pilot_length],pilot_id[0])
                # Correct residual phase rotation
                if correct_phase and phase_drift is not None:
                    self._rx_symbols[next_offset:next_offset+self._mw.pilot_length],last_phase_offset = self._mw.correct_phase_offset (self._rx_symbols[next_offset:next_offset+self._mw.pilot_length],phase_offset,phase_drift)
                    # Check correction
                    print ('[Debug] check phase offset and drift correction with {:d}'.format (pilot_id[0]))
                    corrected_phase_offset,corrected_phase_drift,_ = self._mw.siso_channel_estimation (self._rx_symbols[next_offset:next_offset+self._mw.pilot_length],pilot_id[0])
                # If more pilot ID are specified, run channel estimation on them as well
                if len (pilot_id) > 1:
                    print ('[Debug] extra channel estimation')
                    for k in np.arange (1,len (pilot_id)):
                        self._mw.siso_channel_estimation (self._rx_symbols[next_offset:next_offset+self._mw.pilot_length],pilot_id[1])
                        

            next_offset += self._mw.pilot_length
            # Check whether there is still something to decode
            if len (self._rx_symbols[next_offset:]) > self._payload_length:
                # Correct residual phase rotation
                if correct_phase and phase_drift is not None:
                    self._rx_symbols[next_offset:next_offset+self._payload_length],last_phase_offset = self._mw.correct_phase_offset (self._rx_symbols[next_offset:next_offset+self._payload_length],last_phase_offset,phase_drift)

                # Decode symbols
                self._mw.decode_symbols (self._rx_symbols[next_offset:next_offset+self._payload_length])
                offset = next_offset + self._payload_length
            else:
                # No more symbols: correct residual phase rotation first
                if correct_phase and phase_drift is not None:
                    self._rx_symbols[next_offset:],_ = self._mw.correct_phase_offset (self._rx_symbols[next_offset:],last_phase_offset,phase_drift)

                print ('Partial decoding: {:d} < {:d}'.format (len (self._rx_symbols[next_offset:]),self._payload_length))
                self._mw.decode_symbols (self._rx_symbols[next_offset:])
                more_data = False
            print ('[Debug] {:d}:'.format (k),self._mw.data[:10],self._mw.data[-10:])  # Debug
            k += 1

    def check_periodicity (self):
        if self._mw.check_periodicity ():
            print ('Periodicity check successful.')
        else:
            print ('Periodicity check failed.')

def main ():
    # Test generation and decoding
    mw = measurementSymbols ()
    test_data = mw.generate_symbols (with_preamble=False,with_pilot=False,debug=True)
    mw.decode_symbols (mw.symbols)
    if np.allclose (mw.data,test_data):
        print ('Generation/decoding validated.')
    else:
        print ('Generation/decoding failed')
        sys.exit (-1)
    mw = measurementSymbols (payload_length=2000,periodicity=200,seed=91)
    mw.generate_symbols (with_preamble=False,with_pilot=False)
    mw.decode_symbols (mw.symbols)
    if mw.check_periodicity ():
        print ('Periodicity satisfied.')
    else:
        print ('Periodicity check failed.')
        sys.exit (-1)
    # Test MIMO
    mw = measurementSymbols (mimo_streams=2)
    mw.generate_pilot ()
    if np.dot (mw._pilot[0],mw._pilot[1].conj ())==0:
        print ('Cross-correlation between pilots checked.')
    else:
        print ('Cross-correlation between pilots failed.')
        sys.exit (-1)
    if np.isclose (np.dot (mw._pilot[0],mw._pilot[0].conj ()).real,mw._pilot_root_length*mw._pilot_seq_length):
        print ('Cross-correlation between pilots checked.')
    else:
        print ('Cross-correlation between pilots failed.')
        sys.exit (-1)


if __name__ == '__main__':
    main ()
