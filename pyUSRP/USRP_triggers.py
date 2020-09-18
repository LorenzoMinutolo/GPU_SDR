########################################################################################
##                                                                                    ##
##  THIS LIBRARY IS PART OF THE SOFTWARE DEVELOPED BY THE JET PROPULSION LABORATORY   ##
##  IN THE CONTEXT OF THE GPU ACCELERATED FLEXIBLE RADIOFREQUENCY READOUT PROJECT     ##
##                                                                                    ##
########################################################################################
from .USRP_low_level import *
import numpy as np
from scipy import signal
from .USRP_fitting import get_fit_param
from .USRP_noise import calculate_frequency_timestream
import h5py
import time
class trigger_template(object):
    '''
    Example class for developing a trigger.
    The triggering method has to be passed as an argument of the Packets_to_file function and has to respect the directions given in the Trigger section of this documentation.
    The user is responible for the initialization of the object.
    The internal variable trigger_coltrol determines if the trigger dataset bookeep whenever the trigger method returns metadata['length']>0 or if it's controlled by the user.
    In case the trigger_control is not set on \'AUTO\' the user must take care of expanding the dataset before writing.
    '''

    def __init__(self, rate):
        self.trigger_control = "MANUAL" # OR MANUAL
        if (self.trigger_control != "AUTO") and (self.trigger_control != "MANUAL"):
            err_msg = "Trigger_control in the trigger class can only have MANUAL or AUTO value, not \'%s\'"%str(self.trigger_control)
            print_error(err_msg)
            raise ValueError(err_msg)

    def dataset_init(self, antenna_group):
        '''
        This function is called on file creation an is used to create additional datasets inside the hdf5 file.
        In order to access the datasets created here in the trigger function make them member of the class:

        >>> self.custom_dataset = antenna_group.create_dataset("amazing_dataset", shape = (0,), dtype=np.dtype(np.int64), maxshape=(None,), chunks=True)

        Note: There is no need to bookeep when (at what index) the trigger is called as this is already taken care of in the trigger dataset.
        :param antenna_group is the antenna group containing the 'error','data' and triggering datasets.
        '''

        self.trigger_group = antenna_group['trigger']

        return
    def write_trigger(self, data):

        current_len_trigger = len(self.trigger)
        self.trigger.resize(current_len_trigger+1,0)
        self.trigger[current_len_trigger] = data

    def trigger(self, data, metadata):
        '''
        Triggering function.
        Make modification to the data and metadata accordingly and return them.

        :param data: the data packet from the GPU server
        :param metadata: the metadata packet from the GPU server

        :return same as argument but with modified content.

        Note: the order of data at this stage follows the example ch0_t0, ch1_t0, ch0_t1, ch1_t1, ch0_t2, ch1_t2...
        '''

        return data, metadata


class trigger_example(object):
    '''
    Example class for developing a trigger.
    The triggering method has to be passed as an argument of the Packets_to_file function and has to respect the directions given in the Trigger section of this documentation.
    The user is responible for the initialization of the object.
    The internal variable trigger_coltrol determines if the trigger dataset bookeep whenever the trigger method returns metadata['length']>0 or if it's controlled by the user.
    In case the trigger_control is not set on \'AUTO\' the user must take care of expanding the dataset before writing.
    '''

    def __init__(self, rate = 1):
        self.trigger_control = "MANUAL" # OR MANUAL
        if (self.trigger_control != "AUTO") and (self.trigger_control != "MANUAL"):
            err_msg = "Trigger_control in the trigger class can only have MANUAL or AUTO value, not \'%s\'"%str(self.trigger_control)
            print_error(err_msg)
            raise ValueError(err_msg)

        self.signal_accumulator = 0
        self.std_threshold_multiplier = 10.
        self.trigget_count = 0

    def dataset_init(self, antenna_group):
        '''

        TRIGGER IMPLEMENTATION EXAMPLE

        This function is called on file creation an is used to create additional datasets inside the hdf5 file.
        In order to access the datasets created here in the trigger function make them member of the class:

        >>> self.custom_dataset = antenna_group.create_dataset("amazing_dataset", shape = (0,), dtype=np.dtype(np.int64), maxshape=(None,), chunks=True)

        Note: There is no need to bookeep when (at what index) the trigger is called as this is already taken care of in the trigger dataset.
        :param antenna_group is the antenna group containing the 'error','data' and triggering datasets.
        '''

        # Create a dataset for storing some timing info. This dataset will be updated when the trigger is fired (see write trigger method)
        self.trigger_timing = antenna_group.create_dataset("timing", shape = (0,), dtype=np.dtype(np.float64), maxshape=(None,))

        # Because this is a fancy example we'll store also the triggering threshold and the length of the packet in samples (yes it's dynamically accomodated in the comm protocol)
        self.thresholds = antenna_group.create_dataset("thresholds", shape = (0,), dtype=np.dtype(np.float64), maxshape=(None,))
        self.slices = antenna_group.create_dataset("slices", shape = (0,), dtype=np.dtype(np.int32), maxshape=(None,))

    def write_trigger(self, metadata):

        self.trigget_count +=1

        current_len_trigger = len(self.trigger_timing) # check current length of the dset
        (self.trigger_timing).resize((self.trigget_count,)) # update the length (expensive disk operation ~500 cycles on x86 intel, separate client server if rate>100Msps)
        self.trigger_timing[self.trigget_count-1] = time.time() - self.start_time # write the data of interest

        current_len_trigger = len(self.thresholds)
        (self.thresholds).resize((self.trigget_count,))
        self.thresholds[self.trigget_count-1] = self.signal_accumulator

        current_len_trigger = len(self.slices)
        (self.slices).resize((self.trigget_count,))
        self.slices[self.trigget_count-1] = metadata['length']



    def trigger(self, data, metadata):
        '''
        Triggering function.
        Make modification to the data and metadata accordingly and return them.

        :param data: the data packet from the GPU server
        :param metadata: the metadata packet from the GPU server

        :return same as argument but with modified content.

        Note: the order of data at this stage follows the example ch0_t0, ch1_t0, ch0_t1, ch1_t1, ch0_t2, ch1_t2...

        ***IMPLEMENTATION SPECIFIC***

        This trigger check on the standard deviation of all cthe channels together and write data if the std is bigger than the previous packet.
        This is not ment as a usable object but only as a guideline for designing more functional stuff.

        '''
        if metadata['packet_number'] < 2:
            self.signal_accumulator = np.std(data)
            metadata['length'] = 0
            self.start_time = time.time()
            return [], metadata # This line does not write any data to disk. Note that you can still write the packet and save the trigger info.
        elif self.signal_accumulator*self.std_threshold_multiplier < np.std(data):
            current_time = time.time() - self.start_time
            print("  Triggered at %.2f seconds with a std of %.3e" % (current_time, self.signal_accumulators))
            self.write_trigger(metadata)
            return data, metadata
        else:
            self.signal_accumulator = np.std(data)
            metadata['length'] = 0
            return [], metadata
        
        
class infn(object):
    def __init__(self,rate,threshold,slice_len,noise_trigger=0,mode="pulse",decimation=100,thresholdmode="mag",**kwargs):
        self.rate=rate # realrate
        self.threshold =threshold
        self.stored_data = []
        self.bounds = []
        self.nglitch = []
        self.glitch_indices = [] 
        self.samples_per_packet = []
        self.noise_trigger=noise_trigger
        self.mode=mode #triggerd on pulse/noise or both
        self.slice_len=slice_len
        self.decimation=decimation
        self.start_flag=True
        self.buff_data=[]
        self.buff_time=0.2 #length of each buff, in second
        self.main_buff=[]
        self.buff_full=False
        self.countbuff=0
        self.realpoint=0
        self.numpoint=int(self.slice_len*self.rate) #num of points be recorded
        self.thresholdmode=thresholdmode
        if thresholdmode=="rotate":
            try:
               self.alpha=kwargs['alpha']
            except KeyError:
               print("Need alpha for rotate")
    def buff(self,data,metadata):
        #merge small packets into a big one
        self.noise_mark=[] 
        self.glitch_chan=[]
        self.timestamp=[]
        self.tglitch=[]
        data=np.reshape(data.T,-1)
        self.buff_data=np.append(self.buff_data,data)
        buff_data=self.buff_data
        self.countbuff+=1
        lenbuff=len(self.buff_data)        
        if lenbuff>self.rate*self.buff_time*metadata['channels']:
            self.buff_full=True
            self.main_buff=self.buff_data
            self.buff_data=[]
        return self.buff_full
    def trigger(self,metadata):
        def diff(current,stddev): #return the glitchs position above the threshold
            y=1
            hit_diff=np.ediff1d(current)
            jump=np.where(np.abs(hit_diff)>stddev*self.threshold[ch])
            while (y<len(jump)):
                if jump[y]-jump[y-1]<self.numpoint:
                    jump=np.delete(jump, y)
                    y-=1
                y+=1
            hit_indices=jump
            return hit_indices
        n_chan=metadata['channels']
        self.stored_data=np.reshape(self.main_buff,(-1,n_chan)).T
        self.buff_full=False
        if self.start_flag==True:
            cutoff=int(self.rate*0.05) ## cutoff the beginning of data where contains noise
            self.stored_data=self.stored_data[:,cutoff:]
            self.start_flag=False
        n_samples = self.stored_data.shape[1] ##The number of samples per channel in the packet
        srate = self.rate
        before=int(self.numpoint*0.2)
        after=int(self.numpoint*0.8)
        res=np.zeros(0)        
        
        for x in range(0, n_chan):
            hits = np.zeros(n_samples, dtype=bool)
            ch = int(x)
            if self.thresholdmode=="mag":
                current = np.abs(self.stored_data[ch])
            elif self.thresholdmode=="pha":
                current = np.angel(self.stored_data[ch])
            elif self.thresholdmode=="rotate":
                print("rotate angle is "+str(self.alpha[ch]))
                current =self.stored_data[ch]*(np.cos(self.alpha[ch])+1j*np.sin(self.alpha[ch]))
                current=current.real
                #fired on rotated angle, but original data will be recorded
            hit_noise=[]
            hit_indices=[]
            if self.mode=="noise" or self.mode=="both":
                noise_step=int(self.noise_trigger*srate)
                hit_noise=np.arange(noise_step,len(current),noise_step)
            if self.mode=="pulse" or self.mode=="both":
                med = np.median(current)
                stddev = np.std(current)
                hit_indices=diff(current,stddev)              
            hit_indices=np.append(hit_noise,hit_indices)
            hit_indices=np.sort(hit_indices)
            diff_noise_pulse=np.ediff1d(hit_indices)
            np_close=np.where(diff_noise_pulse<self.numpoint)[0]
            count_chan=0
            n_glitch = len(hit_indices)
            n_noise=len(hit_noise)
            n_pulse=n_glitch-n_noise
            print (str(n_pulse)+" pulse and "+str(n_noise)+" noise detected in chanl "+str(ch)+".")
            for z in range(0, len(hit_indices)): #classify noise and pulse                
                i = hit_indices[z]                
                if i>=before and i<n_samples-after:
                    time_stamp=np.true_divide(i+self.realpoint, srate)                    
                    chopped =self.stored_data[ch,(i-before):(i+after)]                                 
                    if self.mode=="pulse":
                       self.noise_mark.append(0) 
                       self.glitch_chan.append(ch) 
                    else:
                      if int(i/noise_step)*noise_step==i: # when the time match the noise trigger
                           if z>0:
                               if i-hit_indices[z-1]<self.numpoint: # when the pulse drop into noise window                                   
                                   n_glitch = n_glitch - 1                                   
                                   continue
                               else:
                                   self.noise_mark.append(1)
                                   self.glitch_chan.append(ch) 
                           else:
                               self.noise_mark.append(1)
                               self.glitch_chan.append(ch) 
                      else:
                          self.noise_mark.append(0)
                          self.glitch_chan.append(ch) 
                    self.timestamp.append(time_stamp)
                    res=np.append(res,chopped)
                    self.nglitch.append(count_chan)
                    count_chan += 1
                else:
                    pass
                    print ("Glitch index", i, "not in range.")
                    n_glitch = n_glitch - 1                
            self.tglitch.append(n_glitch)
        glitch_total=sum(self.tglitch)
        res=res.reshape([int(glitch_total),self.numpoint]).T
        metadata['channels']=glitch_total
        metadata['length']=int(self.numpoint*glitch_total)
        self.realpoint+=n_samples
        return res,metadata
