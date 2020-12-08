########################################################################################
##                                                                                    ##
##  THIS LIBRARY IS PART OF THE SOFTWARE DEVELOPED BY THE JET PROPULSION LABORATORY   ##
##  IN THE CONTEXT OF THE GPU ACCELERATED FLEXIBLE RADIOFREQUENCY READOUT PROJECT     ##
##                                                                                    ##
########################################################################################

import numpy as np
import scipy.signal as signal
import signal as Signal
import h5py
import sys
import struct
import json
import os
import socket
import queue
from queue import Empty
from threading import Thread,Condition
import multiprocessing
from joblib import Parallel, delayed
from subprocess import call
import time
import gc
import datetime

#plotly stuff
from plotly.graph_objs import Scatter, Layout
from plotly import tools
#import plotly.plotly as py
import plotly.graph_objs as go
import plotly
import colorlover as cl

#matplotlib stuff
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pl
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches

#needed to print the data acquisition process
import progressbar

#import submodules
from .USRP_low_level import *
from .USRP_connections import *
from .USRP_plotting import *
from .USRP_files import *
from .USRP_data_analysis import *
from .USRP_delay import *

def get_fastchirp(rf,start_f,last_f,tx_duration,rx_duration,iterations,threshold,gain,rate,output_filename = None,frontend = 'A',device=0, verbose = False,**kwargs):
    '''
    Perform a fastchirp measure.

    Arguments:
        - start_f: starting frequency of the chirp signal.
        - last_f: last frequency of the fastchirp scan.
        - rf: central LO frequency.
        - device: the on-server device number to use. default is 0.
        - frontend: 'front-end character: A or B. Default is A.
        - tx_duration: Fastchirp transmit time in s.
        - rx_duration: Fastchirp receive time in s.
        - gain: Set the transmission gain. Default 0 dB.
        - iterations: how many fastchirp iteration to perform.
        - output_filename: name of the file. Default is USRP_Fastchirp_timestamp
        - rate: acquisition rate in sps.
        - decimation: decimation parameter TBD on server decim.
        - verbose: verbose output.
    '''

    if not Device_chk(device):
        err_msg = "Something is wrong with the device check in the VNA function."
        print_error(err_msg)
        raise ValueError(err_msg)

    try:
        delay = LINE_DELAY[str(int(rate/1e6))]
        delay *= 1e-9
    except KeyError:
        print_warning("Cannot find associated line delay for a rate of %d Msps. Performance may be negatively affected"%(int(Rate/1e6)))
        delay = 0

    if not Front_end_chk(frontend):
        err_msg = "Cannot detect front_end: "+str(frontend)
        print_error(err_msg)
        raise ValueError(err_msg)
    else:
        TX_frontend = frontend+"_TXRX"
        RX_frontend = frontend+"_RX2"

    if output_filename is None:
        output_filename = "USRP_Fastchirp_"+get_timestamp()
    else:
        output_filename = str(output_filename)


    number_of_samples_tx = rate * tx_duration * iterations
    number_of_samples_rx = rate * (tx_duration + rx_duration) * iterations

    #approximate at 50ks
    tx_buff_len = np.ceil(rate * tx_duration / 50e3) * 50e3
    print_debug("using TX buffer size: %d samples" % tx_buff_len)

    # TBD stuff
    threshold = 1.

    fc_command = global_parameter()

    fc_command.set(TX_frontend,"mode", "TX")
    fc_command.set(TX_frontend,"buffer_len", tx_buff_len)
    fc_command.set(TX_frontend,"gain", gain)
    fc_command.set(TX_frontend,"delay", 1)
    fc_command.set(TX_frontend,"samples", number_of_samples_tx)
    fc_command.set(TX_frontend,"rate", rate)
    fc_command.set(TX_frontend,"bw", 2*rate)

    fc_command.set(TX_frontend,"wave_type", ["CHIRP"])
    fc_command.set(TX_frontend,"ampl", [1.])
    fc_command.set(TX_frontend,"freq", [start_f])
    fc_command.set(TX_frontend,"chirp_f", [last_f])
    fc_command.set(TX_frontend,"swipe_s", [1e9])
    fc_command.set(TX_frontend,"chirp_t", [tx_duration])
    fc_command.set(TX_frontend,"rf", rf)
    fc_command.set(TX_frontend,"burst_on", tx_duration) #should be in samples
    fc_command.set(TX_frontend,"burst_off", rx_duration)

    fc_command.set(RX_frontend,"mode", "RX")
    fc_command.set(RX_frontend,"buffer_len", 1e6)
    fc_command.set(RX_frontend,"gain", 0)
    fc_command.set(RX_frontend,"delay", 1+delay)
    fc_command.set(RX_frontend,"samples", number_of_samples_rx)
    fc_command.set(RX_frontend,"rate", rate)
    fc_command.set(RX_frontend,"bw", 2*rate)

    fc_command.set(RX_frontend,"wave_type", ["NODSP"])
    fc_command.set(RX_frontend,"ampl", [1.0]) # amy act as threshold in future
    fc_command.set(RX_frontend,"freq", [start_f])
    fc_command.set(RX_frontend,"chirp_f", [last_f])
    fc_command.set(RX_frontend,"swipe_s", [1e9])
    fc_command.set(RX_frontend,"chirp_t", [number_of_samples_rx])
    fc_command.set(RX_frontend,"rf", rf)
    fc_command.set(RX_frontend,"decim", threshold) # THIS only activate the decimation.

    measure_complete = False

    if fc_command.self_check():
        if(verbose):
            print("Fastchirp command succesfully checked")
            fc_command.pprint()

        Async_send(fc_command.to_json())

    else:
        print_warning("Something went wrong with the setting of VNA command.")
        return ""

    time.sleep(1) # mimic the std delay on the server.

    Packets_to_file(
        parameters = fc_command,
        timeout = None,
        filename = output_filename,
        dpc_expected = number_of_samples_rx,
        meas_type = "fastchirp", **kwargs
    )

    print_debug("Fastchirp acquisition terminated.")


    return output_filename

def analyze_fastchirp():
    '''
    Analyze a fastchirp measure: TBD. Debug tool for now.
    '''
    return
