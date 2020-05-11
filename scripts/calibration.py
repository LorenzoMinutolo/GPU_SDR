import sys,os,glob,shutil
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

    TOO_LOW_LOOPBACK = 20
    TOO_HIGH_LOOPBACK = 40

    parser = argparse.ArgumentParser(description='Calibrate the USRP server filters and the py_sdr library calibration by using loopback configuration. Suggested loopback attenuation is between %d and %d dB. For more information refer to the documentation.' % (TOO_LOW_LOOPBACK, TOO_HIGH_LOOPBACK))

    parser.add_argument('--folder', '-fn', help='Name of the main folder in which the data are stored', type=str, default = "data")
    parser.add_argument('--sub_folder', '-sf', help='Name of the subfolder in which the calibration data are stored', type=str, default = "calibration")
    parser.add_argument('--loopback', '-l', help='Value in dB for total loopback attenuation', type=float, required=True)
    parser.add_argument('--gain', '-g', help='TX gain value in dB', type=int, default=0)

    #will be removed when a full Dboard calibration is in place
    parser.add_argument('--freq', '-f', help='Central frequency where to calibrate', type=float, required=True)

    # default should be changed to calibrate both cards. This will imply setting two different calibration constants and loopback attenuation values, TODO
    parser.add_argument('--frontend', '-rf', help='front-end character: A or B', type=str, default="A")


    args = parser.parse_args()

    #Warning about too low attenuation
    if args.loopback < TOO_LOW_LOOPBACK:
        u.print_warning("loopback attenuation is too low, suggeested min value is %d dB, signal distortion may be happen in the digitization" % TOO_LOW_LOOPBACK)


    #warning about too high attenuation
    if args.loopback > TOO_HIGH_LOOPBACK:
        u.print_warning("loopback attenuation is too high, suggeested max value is %d dB, SNR may negatively affect calibration precision" % TOO_HIGH_LOOPBACK)

    # Navigate to the derised folder
    try:
        os.mkdir(args.folder)
    except OSError:
        pass

    os.chdir(args.folder)


    try:
        os.mkdir(args.sub_folder)
    except OSError:
        u.print_warning("Removing previous calibration")
        try:
            shutil.rmtree(args.sub_folder)
        except OSError:
            raise ValueError("Cannot remove original calibration folder!")

        os.mkdir(args.sub_folder)

    os.chdir(args.sub_folder)

    # Connect to USRP server
    if not u.Connect():
        exit()

    # Set the variables for the scans

    # Should be retrived from server properties, communication scheme not implemented yet
    DCARD_ANALOG_BW = 20e6

    # Should as well be setted with a bechmark_rate-like automatic program
    SYSTEM_MAX_RATE = 20e6

    # This could become a user argument
    FILTER_LENGTH = 10000

    # IF bandwith related. Again a sort of wizard could be implemented to determine the optimal value starting from resonator Q's.
    MEASURE_TIME = 10

    # Derived from signal to noise ratio but not relevant in a client library calibration
    ITERATIONS_NUMBER = 5

    # Set the central frequency for the scan TODO: hsoul also check range compatibiulity with Dcard information
    CENTRAL_FREQUENCY = args.freq * 1e6

    # Semispan around central frequency in which the S21 is flat
    FLAT_HALF_BW = 10e6

    # Semispan around center spike
    SPIKE_HALF_BW = 100e3

    # Set the line delay. This may also become part of the calibration (?)

    delay_duration = 0.01

    try:
        if u.LINE_DELAY[str(int(SYSTEM_MAX_RATE/1e6))]: pass
    except KeyError:

        print("Cannot find line delay. Measuring line delay before VNA:")

        delay_filename = u.measure_line_delay(
            SYSTEM_MAX_RATE, CENTRAL_FREQUENCY, args.frontend,
            USRP_num=0, tx_gain=0, rx_gain=0, output_filename=None, compensate = True,
            duration = delay_duration
        )

        delay = u.analyze_line_delay(delay_filename, True)

        u.write_delay_to_file(delay_filename, delay)

        u.load_delay_from_file(delay_filename)


    # Perform a VNA
    vna_filename = u.Single_VNA(
        start_f = -DCARD_ANALOG_BW/2,
        last_f = DCARD_ANALOG_BW/2,
        measure_t = MEASURE_TIME,
        n_points = FILTER_LENGTH,
        tx_gain = args.gain,
        Rate=SYSTEM_MAX_RATE,
        decimation=True,
        RF=CENTRAL_FREQUENCY,
        Front_end=args.frontend,
        Device=None,
        output_filename=None,
        Multitone_compensation=None,
        Iterations=ITERATIONS_NUMBER,
        verbose=False
    )

    # Maybe one day this script will pass the file to the server and upload filters window
    u.Disconnect()




    # Set the calibration library constant to 1. TODO: remove this when a full calibration system is in place
    u.change_calibration(1.)

    #DEBUG!!!
    #print(u.get_VNA_calib(vna_filename))

    # Analyze the VNA
    u.VNA_analysis(vna_filename)

    # Plot the uncalibrated VNA
    u.plot_VNA(vna_filename, backend = "matplotlib", output_filename = "uncalibrated_VNA")

    '''
    Debug part, in future a USRP_calibration file will be produced
    '''


    CENTRAL_FREQUENCY = args.freq * 1e6

    # Semispan around central frequency in which the S21 is flat
    FLAT_HALF_BW = 10e6

    # Semispan around center spike
    SPIKE_HALF_BW = 500e3

    #calculation for the right calib constant
    freq, S21 =  u.get_VNA_data(vna_filename, calibrated = True, usrp_number = 0)
    baseband_freq_abs = np.abs(freq - CENTRAL_FREQUENCY)
    select_vector = np.ones(len(freq)) #np.logical_and(baseband_freq_abs<FLAT_HALF_BW , baseband_freq_abs>SPIKE_HALF_BW)
    print("Excluding first and last %d points."%int(len(select_vector)*0.2))
    for i in range(int(len(select_vector)*0.2)):
            select_vector[i] = 0
            select_vector[-i] = 0
    select_vector = np.asarray(select_vector, dtype=bool)
    print(select_vector)
    current_level = u.linear2db(np.abs(np.mean(S21[select_vector])))
    difference = -(current_level + args.loopback)
    calculated_calibration = u.db2linear(difference)

    #set the calibration constant
    u.change_calibration(calculated_calibration)

    #reanalize the VNA
    u.VNA_analysis(vna_filename)
    u.plot_VNA(vna_filename, backend = "matplotlib", output_filename = "calibrated_VNA")
    print(("\n\nCalibration is %f\n\n" % calculated_calibration))
