import sys,os,time
import numpy as np
try:
    import pyUSRP as u
except ImportError:
    try:
        sys.path.append('..')
        import pyUSRP as u
    except ImportError:
        print("Cannot find the pyUSRP package")

def set_dc(V):
    cmd =  "lxi scpi --address 10.11.9.230 \"C1:BSWV WVTP,DC\""
    os.system(cmd)
    time.sleep(0.2)
    cmd =  "lxi scpi --address 10.11.9.230 \"C1:BSWV OFST,%.3f\"" % V
    os.system(cmd)

def set_ac(V):
    cmd =  "lxi scpi --address 10.11.9.230 \"C1:BSWV WVTP,SQUARE\""
    os.system(cmd)
    time.sleep(0.2)
    cmd =  "lxi scpi --address 10.11.9.230 \"C1:BSWV FRQ,1\""
    os.system(cmd)
    time.sleep(0.2)
    cmd =  "lxi scpi --address 10.11.9.230 \"C1:BSWV OFST,%.3f\"" % V
    os.system(cmd)
    time.sleep(0.2)
    cmd =  "lxi scpi --address 10.11.9.230 \"C1:BSWV AMP,0.07\""
    os.system(cmd)


if __name__ == "__main__":

    if not u.Connect():
        u.print_error("Cannot find the GPU server!")
        exit()

    os.chdir("data")
    os.mkdir("swipe")
    os.chdir("swipe")

    filename = u.measure_line_delay(200e6, 270e6, 'A', USRP_num=0, tx_gain=0, rx_gain=0, output_filename=None, compensate = True, duration = 0.1)

    delay = u.analyze_line_delay(filename, False)

    u.write_delay_to_file(filename, delay)

    u.load_delay_from_file(filename)

    for gain in [10,20,30]:
        for dc in np.arange(15)/9.:
            custom_name_VNA = "USRP_VNA_DC%d_G%d" % (dc*1000, gain)
            custom_name_calib = "USRP_Calib_DC%d_G%d" % (dc*1000, gain)
            custom_name_noise = "USRP_Noise_DC%d_G%d" % (dc*1000, gain)

            set_dc(dc)
            err = True
            while(err):
                vna_filename = u.Single_VNA(
                    start_f = -45e6,
                    last_f = 25e6,
                    measure_t = 60,
                    n_points = 100000,
                    tx_gain = gain,
                    Rate=200e6,
                    decimation=True,
                    RF=269.999999e6,
                    Front_end='A',
                    Device=None,
                    output_filename=custom_name_VNA,
                    Multitone_compensation=7,
                    Iterations=1,
                    verbose=False
                )
                err = u.check_errors(vna_filename)
                if err:
                    os.remove(vna_filename+".h5")
            u.VNA_analysis(vna_filename)
            # u.plot_VNA(vna_filename, backend = 'plotly', unwrap_phase = True)
            u.plot_VNA(vna_filename, backend = 'matplotlib', unwrap_phase = True)
            th = 0.8
            sm = 10
            x = []
            while len(x) <6:
                print("settign threshold to %f" % th)
                x = u.extimate_peak_number(vna_filename, threshold = th, smoothing = sm, peak_width = 0.3e6, verbose = False, exclude_center = True, diagnostic_plots = False)
                th -= 0.001
                if th<0:
                    th = 0.8
                    sm -= 2
                    if sm < 1:
                        print("FITTING FAILED")
                        exit()
            print(x)
            u.vna_fit(vna_filename, p0=None, fit_range = 0.2e6/3, verbose = False)
            u.plot_resonators([vna_filename], reso_freq = None, backend = 'matplotlib', title_info = None, verbose = False, output_filename = None, auto_open = True, attenuation = None,single_plots = False)


            guard_tones = [5e6,]
            rf_freq, tones = u.get_tones(vna_filename)
            tones = np.concatenate((tones,guard_tones))
            err = True
            while(err):
                noise_filename = u.get_tones_noise(
                    tones,
                    measure_t = 90,
                    rate = 200e6,
                    decimation = 1000,
                    amplitudes = [1./7. for i in range(7)],
                    RF = 269.999999e6,
                    output_filename = custom_name_noise,
                    Front_end = 'A',
                    Device = None,
                    delay = None,
                    pf_average = 8,
                    tx_gain = gain,
                    mode = 'DIRECT',
                    trigger = None,
                    shared_lo = False
                )
                err = u.check_errors(noise_filename)
                if err:
                    os.remove(noise_filename+".h5")
            u.copy_resonator_group(vna_filename, noise_filename)


            set_ac(dc)
            err = True
            while(err):
                noise_filename = u.get_tones_noise(
                    tones,
                    measure_t = 60,
                    rate = 200e6,
                    decimation = 4000,
                    amplitudes = [1./7. for i in range(7)],
                    RF = 269.999999e6,
                    output_filename = custom_name_calib,
                    Front_end = 'A',
                    Device = None,
                    delay = None,
                    pf_average = 8,
                    tx_gain = gain,
                    mode = 'DIRECT',
                    trigger = None,
                    shared_lo = False
                )
                err = u.check_errors(noise_filename)
                if err:
                    os.remove(noise_filename+".h5")
            #u.plot_raw_data([noise_filename], decimation=100, low_pass=5, backend='plotly', output_filename=None, mode='PM', auto_open=True, end_time = 3)
            u.copy_resonator_group(vna_filename, noise_filename)
            u.plot_frequency_timestreams([noise_filename], decimation=100, low_pass=5, backend='plotly', output_filename=None, auto_open=False, end_time=10)
            u.plot_frequency_timestreams([noise_filename], decimation=100, low_pass=5, backend='matplotlib', output_filename=None, auto_open=True, end_time=10)
