
import sys,os
import numpy as np
try:
    import pyUSRP as u
except ImportError:
    try:
        sys.path.append('..')
        import pyUSRP as u
    except ImportError:
        print("Cannot find the pyUSRP package")

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test the basic VNA functionality.')

    parser.add_argument('--folder', '-fn', help='Name of the folder in which the data will be stored', type=str, default = "data")
    parser.add_argument('--freq', '-f', help='LO frequency in MHz. Default 300 MHz', type=float)
    parser.add_argument('--rate', '-r', help='Sampling frequency in Msps', type=float, default = 100)
    parser.add_argument('--frontend', '-rf', help='front-end character: A or B. Default is A', type=str, default="A")
    parser.add_argument('--f0', '-f0', help='Baseband start frequrency in MHz. Default is full bandwidth available.', type=float)
    parser.add_argument('--f1', '-f1', help='Baseband end frequrency in MHz. Default is full bandwidth available.', type=float)
    parser.add_argument('--time', '-t', help='Total duration of the measurement. Including transmit and receive phases.', type=float, default=10)
    parser.add_argument('--t_tx', '-ttx', help='Fastchirp transmit time in ms', type=float, default=10)
    parser.add_argument('--t_rx', '-trx', help='Fastchirp receive time in ms', type=float, default=50)
    parser.add_argument('--gain', '-g', help='Set the transmission gain. Default 0 dB',  type=int, default=0)
    parser.add_argument('--delay_over', '-do', help='Manually set line delay in nanoseconds. Skip the line delay measure.',  type=float)
    parser.add_argument('--threshold', '-th', help='TBD online analysis parameter',  type=int, default=0)


    args = parser.parse_args()

    try:
        os.mkdir(args.folder)
    except OSError:
        pass

    os.chdir(args.folder)

    central_freq = int(args.freq*1e6)
    rate = int(args.rate*1e6)

    if args.f0 is None:
        f0 = 1e6*(args.rate/2. - 1)
        u.print_debug("Setting f0 to %.2fMHz" % (f0/1e6))
    else:
        f0 = 1e6*args.rate

    if args.f1 is None:
        f1 = 1e6*(args.rate/2. - 1)
        u.print_debug("Setting f1 to %.2fMHz" % (f1/1e6))
    else:
        f1 = 1e6*args.rate

    if args.t_rx < 0.1:
        u.print_warning("Receiving phase time (%.2f) too low!" % args.t_rx)
        exit(-1)
    else:
        t_rx = args.t_rx * 1e-3

    if args.t_tx < 0:
        u.print_warning("Transmitting phase time (%.2f) cannot be < 0!" % args.t_tx)
        exit(-1)
    else:
        t_tx = args.t_tx * 1e-3

    iterations = np.ceil(float(args.time)/(t_tx+t_rx))
    u.print_debug("Iterating single fastchirp %d times." % iterations)

    if not u.Connect():
        u.print_error("Cannot find the GPU server!")
        exit()

    if args.delay_over is None:
        print("Cannot find line delay. Measuring line delay before starting fastchirp:")
        u.print_warning("This step is not strictly necessary. Will remove in future commits.")

        filename = u.measure_line_delay(rate, central_freq, args.frontend, USRP_num=0, tx_gain=args.gain, rx_gain=0, output_filename=None, compensate = True, duration = 0.2)

        delay = u.analyze_line_delay(filename, False)

        u.write_delay_to_file(filename, delay)

        u.load_delay_from_file(filename)

    else:

        u.set_line_delay(rate, args.delay_over)


    fastchirp_filename = u.get_fastchirp(
        rf=central_freq,
        start_f = f0,
        last_f = f1,
        tx_duration = t_tx,
        rx_duration = t_rx,
        iterations = iterations,
        threshold = args.threshold,
        gain = args.gain,
        rate = rate,
        frontend = args.frontend,
        device=0
    )
    print("Data acquisition done. Filename: %s" % fastchirp_filename)
