import numpy as np
from numpy import sqrt, log10, fft
from tqdm import tqdm
from nptdms import TdmsFile
import matplotlib as mpl
from matplotlib import pyplot as plt
mpl.use('Qt5Agg') #Defines plotting backend
import math
if __name__ == '__main__':
    AverageAmplitude = []
    dbmAmplitude = []
    fft_peak = []
    do_fft = False
    bits_to_volts = (0.1/128)
    two_sqrt_2 = 2*math.sqrt(2)
    tdms_file = TdmsFile.read("NORA_ch0.tdms") #Specify file with data from calibration

    for group in tdms_file.groups():
        group_name = group.name
        for channel in tqdm(group.channels()):
            data = channel[:-1]
            data = data * bits_to_volts # Convert to volts (0.1V/128 bits = 0.00078125)

            if do_fft:
                fft_data = fft.fft(data)
                fft_peak.append(np.max(fft_data))

            rms = ((np.max(data)-np.min(data))/(two_sqrt_2))
            if(rms == 0):
                rms = 0.028
            dbmAmplitude.append(10 * log10(np.abs(rms**2/50))+30)

    dbmAmplitude = np.array(dbmAmplitude) #Convert to numpy array

    x = np.arange(10, 2500, 2)  # Frequencies in MHz
    plt.plot(x[:-1],dbmAmplitude[:-1])
    plt.title("Calibration data for measurement setup at -10dB")
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Amplitude [dBm]")
    plt.show()




