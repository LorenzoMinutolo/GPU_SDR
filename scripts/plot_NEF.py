
import sys,os,glob

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

    parser.add_argument('--folder', '-fn', help='Name of the folder in which the data are stored', type=str, default = "data")
    parser.add_argument('--backend', '-b', help='backend to use for plotting', type=str, default= "matplotlib")
    parser.add_argument('--welch', '-w', help='In how many seg to divide the timestreams', type=int, default= 1)
    parser.add_argument('--att', '-a', help='Line attenuation', type=int, default= 0)
    parser.add_argument('--calib', '-c', help='User calibration in Hz/pW. 0 in the array turns off the channel', nargs='+')
    parser.add_argument('--maxf', '-f', help='Max plotted freq', type=float, default= 1e5)

    args = parser.parse_args()

    try:
        os.mkdir(args.folder)
    except OSError:
        pass

    os.chdir(args.folder)
    if args.calib is not None:
        calib = [float(i) for i in args.calib]
    else:
        calib = None
    files = glob.glob("USRP_Noise*.h5")
    # print(files)
    u.calculate_NEF_spectra(files[0], welch = args.welch, clip = 0.1, verbose = True, usrp_number = 0)
    u.plot_NEF_spectra(files[0], channel_list=None, max_frequency=args.maxf, title_info=None, backend=args.backend,
                        cryostat_attenuation=args.att, auto_open=True, calib = calib, output_filename=None)
