
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

def run(backend, files, decim, att, title):
    for f in files:
        u.VNA_analysis(f)
    u.plot_VNA(files, backend = backend, plot_decim = decim, unwrap_phase = True, att = att, title = title,
        # add_info = [
        #     "",
        #     "",
        #     ""
        # ]

    )



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test the basic VNA functionality.')

    parser.add_argument('--folder', '-fn', help='Name of the folder in which the data are stored', type=str, default = "data")
    parser.add_argument('--backend', '-b', help='backend to use for plotting', type=str, default= "matplotlib")
    parser.add_argument('--title', '-t', help='Title of the plot', type=str, default= None)
    parser.add_argument('--plot_decimate', '-d', help='deciamte data in the plot to get lighter files', type=int, default= None)
    parser.add_argument('--line_attenuation', '-a', help='attenuation befor chip: will display power on chip in plot legend', type=float, default= None)


    args = parser.parse_args()
    os.chdir(args.folder)

    files = glob.glob("USRP_VNA*.h5")

    run(backend = args.backend, files = files, decim = args.plot_decimate, att = args.line_attenuation, title = args.title)
