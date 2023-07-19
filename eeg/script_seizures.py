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

raw = mne.io.read_raw_edf('data/chb05_06.edf')
data = raw.get_data().T
data_sei_06 = data[(417*256):(532*256),:] # seizure
print(data_sei_06)

data_mat = []

for i in range(0, data_sei_06.shape[0]):
    data_mat.append(data_sei_06[i,:])

print(np.array(data_mat).shape)

raw = mne.io.read_raw_edf('data/chb05_13.edf')
data = raw.get_data().T
data_sei_06 = data[(1086*256):(1196*256),:] # seizure

for i in range(0, data_sei_06.shape[0]):
    data_mat.append(data_sei_06[i,:])

print(np.array(data_mat).shape)

raw = mne.io.read_raw_edf('data/chb05_16.edf')
data = raw.get_data().T
data_sei_06 = data[(2317*256):(2413*256),:] # seizure

for i in range(0, data_sei_06.shape[0]):
    data_mat.append(data_sei_06[i,:])

print(np.array(data_mat).shape)

raw = mne.io.read_raw_edf('data/chb05_17.edf')
data = raw.get_data().T
data_sei_06 = data[(2451*256):(2571*256),:] # seizure

for i in range(0, data_sei_06.shape[0]):
    data_mat.append(data_sei_06[i,:])

print(np.array(data_mat).shape)

raw = mne.io.read_raw_edf('data/chb05_22.edf')
data = raw.get_data().T
data_sei_06 = data[(2348*256):(2465*256),:] # seizure

for i in range(0, data_sei_06.shape[0]):
    data_mat.append(data_sei_06[i,:])

data = np.array(data_mat)
print(np.array(data_mat).shape)
m = 5
N = int(data.shape[0] / (256*m))
data_mat = np.zeros([N,23])
for j in range(int(N)):
        index = np.arange(256*m*j,(256*m*(j+1)))
        data_sub = np.max(np.absolute(data[index]),0)
        data_mat[j] = data_sub
        #print(data_sub.shape)

print(data_mat)

fig, ax = plt.subplots()
ax.plot(data_mat)
plt.show()
header = ','.join(raw.ch_names)
np.savetxt('eeg_seizures.csv', data_mat, delimiter = ',', header=header)