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

if __name__ == "__main__":
    meas_dict = [{
        'gain':None,
        'vdc':None
    } for ]
