
import sys,os,random
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

def run(rate,freq,front_end, tones, lapse, decimation, gain, vna, mode, pf, trigger, amplitudes, shared_LO):

    if trigger is not None:
        try:
            trigger = eval('u.'+trigger+'()')
        except SyntaxError:
            u.print_error("Cannot find the trigger \'%s\'. Is it implemented in the USRP_triggers module?"%trigger)
            return ""
        except AttributeError:
            u.print_error("Cannot find the trigger \'%s\'. Is it implemented in the USRP_triggers module?"%trigger)
            return ""

    if shared_LO is None:
        shared_LO = False
    #trigger = u.trigger_template(rate = rate/decimation)
    noise_filename = u.get_tones_noise(tones, measure_t = lapse, rate = rate, decimation = decimation, amplitudes = amplitudes,
                              RF = freq, output_filename = None, Front_end = front_end,Device = None, delay = None,
                              pf_average = pf, tx_gain = gain, mode = mode, trigger = trigger, shared_lo = shared_LO)
    if vna is not None:
        u.copy_resonator_group(vna, noise_filename)

    return noise_filename


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test the basic Noise acquisition functionality')

    parser.add_argument('--folder', '-fn', help='Name of the folder in which the data will be stored. Move to this folder before everything', type=str, default = "data")
    parser.add_argument('--freq', '-f', help='LO frequency in MHz', type=float, default= 300)
    parser.add_argument('--gain', '-g', help='TX noise', type=int, default= 0)
    parser.add_argument('--rate', '-r', help='Sampling frequency in Msps', type=float, default = 100)
    parser.add_argument('--frontend', '-rf', help='front-end character: A or B', type=str, default="A")
    parser.add_argument('--tones','-T', nargs='+', help='Tones in MHz as a list i.e. -T 1 2 3')
    parser.add_argument('--guard_tones','-gt', nargs='+', help='Add guard Tones in MHz as a list i.e. -T 1 2 3')
    parser.add_argument('--decimation', '-d', help='Decimation factor required', type=float, default=100)
    parser.add_argument('--time', '-t', help='Duration of the scan in seconds', type=float, default=10)
    parser.add_argument('--pf', '-pf', help='Polyphase averaging factor for PF mode and taps multiplier for DIRECT mode FIR filter', type=int, default=4)
    parser.add_argument('--VNA', '-vna', help='VNA file containing the resonators. Relative to the specified folder above.', type=str)
    parser.add_argument('--mode', '-m', help='Noise acquisition kernels. DIRECT uses direct demodulation PFB use the polyphase filter bank technique.', type=str, default= "DIRECT")
    parser.add_argument('--random', '-R', help='Generate random tones for benchmark and test reason', type=int)
    parser.add_argument('--trigger', '-tr', help='String describing the trigger to use. Default is no trigger. Use the name of the trigger classes defined in the trigger module with no parenthesis', type=str)
    parser.add_argument('--DAC_division', '-dd', help='Divide the DAC to use only 1./this for each tone. Must be >= len(tones).', type=int, default=None)
    parser.add_argument('--delay', '-dy', help='Optional delay file where to souce delay information', type=str)
    parser.add_argument('--addr', '-addr', help='Addredd of the server', type=str, default = None)
    parser.add_argument('--shared_lo', '-slo', help='Enable the shared TX/RX LO. Works only on platforms with export/import LO phyusical ports', action='store_true')


    args = parser.parse_args()
    try:
        os.mkdir(args.folder)
    except OSError:
        pass

    os.chdir(args.folder)

    if args.delay is not None:
        dealy_filename = u.format_filename(args.delay)
        u.load_delay_from_file(dealy_filename)

    if args.VNA is not None:
        rf_freq, tones = u.get_tones(args.VNA)
        u.print_debug("getting %d tones from %s" % (len(tones),args.VNA))
    else:
        if args.random is not None:
            tones = [random.randint(-args.rate/2.,-args.rate/2.) for c in range(args.random)]
        else:
            try:
                if args.tones is not None:
                    tones = [float(x) for x in args.tones]
                    tones = np.asarray(tones)*1e6
                else:
                    tones = []
            except ValueError:
                try:
                    tones = [float(c) for c in args.tones[0].split(" ")]
                    tones = np.asarray(tones)*1e6
                except:
                    u.print_error("Cannot convert tone argument.")

    rf_freq = args.freq*1e6

    if args.random is not None:
        tones = [random.uniform(-args.rate*1e6/2, args.rate*1e6/2) for ui in range(args.random)]

    if not u.Connect(addrss = args.addr):
        u.print_error("Cannot find the GPU server!")
        exit()

    if args.guard_tones is not None:
        guard_tones = [float(x) for x in args.guard_tones]
        guard_tones = np.asarray(guard_tones)*1e6
        # it's important that guard tones are at the end of the tone array !!!
        tones = np.concatenate((tones,guard_tones))

    if args.DAC_division is None:
        args.DAC_division = 1

    if args.DAC_division  >= len(tones):
        amplitudes = [1./args.DAC_division for x in tones]
        print("Amplitudes adjusted to ", amplitudes)
    else:
        u.print_error("Cannot use 1./%d DAC each for %d tones." % (args.DAC_division,len(tones)))
        amplitudes=None

    # Data acquisition

    f = run(rate = args.rate*1e6, freq = rf_freq, front_end = args.frontend,
            tones = np.asarray(tones), lapse = args.time, decimation = args.decimation,
            gain = args.gain, vna= args.VNA, mode = args.mode, pf = args.pf, trigger = args.trigger, amplitudes=amplitudes,
            shared_LO = args.shared_lo)

    # Data analysis and plotting will be in an other python script
