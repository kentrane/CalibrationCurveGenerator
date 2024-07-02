import time
import numpy as np
from numpy import sqrt, log10, fft
from scipy.signal import savgol_filter
from tqdm import tqdm
from nptdms import TdmsFile
import matplotlib as mpl
from matplotlib import pyplot as plt
mpl.use('Qt5Agg') #Defines plotting backend
import math


channel = 0
calibrationfile = f'NORA_ch{channel}.tdms'
def loadcalibration(calibrationfile):
    with TdmsFile.open(calibrationfile) as file:
        data = []
        properties = file['data']['pulse number 1'].properties
        for pulse in tqdm(file['data'].channels(),
                          desc=f'loading data from file {calibrationfile}',
                          unit=" pulses",
                          unit_scale=True):
            data.append(pulse[:])
    return data, properties


if __name__ == '__main__':
    AverageAmplitude = []
    dbmAmplitude = []
    fft_peak = []
    do_fft = False # Don't spend the time if you don't need
    bits_to_volts = (0.1/128)  # Constant since NORA is 8 bit resolution and set to 100mV range
    two_sqrt_2 = 2*math.sqrt(2)  # Constant for faster calculation
    cal_data, properties = loadcalibration(calibrationfile)
    for channel in tqdm(cal_data, desc="Processing", unit=" pulses", unit_scale=True):
        data = channel[:]
        data = data * bits_to_volts  # Convert to volts (0.1V/128 bits = 0.00078125)

        if do_fft:
            fft_data = fft.fft(data)
            fft_peak.append(np.max(fft_data))

        rms = ((np.max(data)-np.min(data))/(two_sqrt_2))
        if(rms == 0):  # small hack to avoid taking log(0)...
            rms = 0.028
        dbmAmplitude.append(10 * log10(np.abs(rms**2/50))+30)  # Convert to dBm at Z0=50 ohm
    print("done!")

    print("plotting")
    dbmAmplitude = np.array(dbmAmplitude)  # Convert to numpy array
    np.save("CalibrationData.npy", dbmAmplitude)
    filtered_response = savgol_filter(dbmAmplitude, 37, 7)
    np.save("CalibrationFiltered.npy", filtered_response)
    # Plot the calibration curve with the frequencies we set during the setup of the calibration
    # Plotting frequency vs amplitude in dBm and remove last sample
    x = np.arange(10, 2500, 2)  # Frequencies in MHz
    plt.title(f'Response for {calibrationfile}')
    plt.plot(x[:-1], dbmAmplitude[:-1], linestyle="none", marker=".", label="unfiltered response")
    plt.plot(x[:-1],filtered_response[:-1], linestyle="-", label="filtered response")
    plt.legend()
    plt.grid()
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Amplitude [dBm]")
    plt.savefig('Response.png', dpi=600)
    plt.show()





