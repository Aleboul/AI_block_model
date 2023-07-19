"""
Compute block maxima (variable denoted by m) for the given chb-mit eeg database.

All signals were sampled at 256 samples per second with 16-bit resolution.
Most files contain 23 EEG signals (24 or 26 in a few cases). The International
10-20 system of EEG electrode positions and nomenclature was used for these
recordings.

The database considered here is the eeg without seizures.

Return a csv file containing the time series of block maxima.
"""

import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_mat = []
for i in range(0, 39):
    print(i)
    if i < 9:
        raw = mne.io.read_raw_edf('data/chb05_0' + str(i+1) + '.edf')
    else:
        raw = mne.io.read_raw_edf('data/chb05_' + str(i+1) + '.edf')
    if (i+1) == 6:
        data = raw.get_data().T
        data = np.delete(data, range(417*256, 532*256), axis=0) # seizure
    if (i+1) == 13:
        data = raw.get_data().T
        data = np.delete(data, range(1086*256, 1196*256), axis=0) # seizure
    if (i+1) == 16:
        data = raw.get_data().T
        data = np.delete(data, range(2317*256, 2413*256), axis=0) # seizure
    if (i+1) == 17:
        data = raw.get_data().T
        data = np.delete(data, range(2451*256, 2413*256), axis=0) # seizure
    if (i+1) == 22:
        data = raw.get_data().T
        data = np.delete(data, range(2348*256, 2465*256), axis=0) # seizure
    else:
        data = raw.get_data().T
    data = np.absolute(data)
    m = 240  # block's length
    N = int(data.shape[0] / (256*m))
    for j in range(int(N)):
        index = np.arange(256*m*j, (256*m*(j+1)))
        data_sub = np.max(data[index], 0)
        data_mat.append(data_sub)
data_mat = np.array(data_mat)
fig, ax = plt.subplots()
ax.plot(data_mat[:, 14])
plt.show()

fig, ax = plt.subplots()
ax.plot(data_mat)
plt.show()
header = ','.join(raw.ch_names)
np.savetxt('eeg_wo_seizures.csv', data_mat, delimiter=',', header=header)

print(data_mat.shape)
