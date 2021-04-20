
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

def run(backend, files, welch, dbc, att, max_f, ch):
    for f in files:
        u.calculate_noise(f, verbose = True, welch = max(welch,1), dbc = dbc, clip = 0.1)
        pass
    print(u.plot_noise_spec(files, channel_list=ch, max_frequency=max_f, title_info=None, backend=backend,
                    cryostat_attenuation=att, auto_open=True, output_filename=None, add_info = None))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Analyze and plot the noise')

    parser.add_argument('--folder', '-fn', help='Name of the folder in which the data are stored', type=str, default = "data")
    parser.add_argument('--backend', '-b', help='backend to use for plotting', type=str, default= "matplotlib")
    parser.add_argument('--welch', '-w', help='Whelch factor relative to timestream length so that welch factor is len(timestream)/this_arg', type=int, default= 5)
    parser.add_argument('--dbc', '-dbc', help='Analyze and plot in dBc or not', action="store_true")
    parser.add_argument('--att', '-a', help='Total attenuation', type=int, default= 0)
    parser.add_argument('--max_f', '-f', help='Max frequency to plot in Hz', type=int, default= 10000)
    parser.add_argument('--channels','-ch', nargs='+', help='only plot certain channels. As a list i.e. -ch 0 1 2 3')

    args = parser.parse_args()

    try:
        os.mkdir(args.folder)
    except OSError:
        pass

    os.chdir(args.folder)

    files = glob.glob("USRP_Noise*.h5")
    ch = [int(x) for x in args.channels]

    run(backend = args.backend, files = files, welch = args.welch, dbc = args.dbc, att = args.att, max_f = args.max_f, ch = ch)
