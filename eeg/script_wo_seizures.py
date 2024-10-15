"""
Compute block maxima (variable denoted by m) for the given chb-mit eeg database.

All signals were sampled at 256 samples per second with 16-bit resolution.
Most files contain 23 EEG signals (24 or 26 in a few cases). The International
10-20 system of EEG electrode positions and nomenclature was used for these
recordings.

The database considered here is the eeg without seizures.

Return a csv file containing the time series of block maxima.
"""

# Import necessary libraries for EEG data processing and visualization
import mne  # MNE for reading and processing EEG/MEG data
import matplotlib.pyplot as plt  # For plotting
import numpy as np  # For numerical computations

# Initialize an empty list to store the processed EEG data
data_mat = []

# Loop through 39 EDF files (from 'chb05_01.edf' to 'chb05_39.edf')
for i in range(0, 39):
    print(i)  # Print the current file index for tracking progress
    
    # Load EEG data from EDF files
    # File naming convention: files 'chb05_01' to 'chb05_09' have a leading zero in their name
    if i < 9:
        raw = mne.io.read_raw_edf('data/chb05_0' + str(i+1) + '.edf')  # e.g., 'chb05_01.edf'
    else:
        raw = mne.io.read_raw_edf('data/chb05_' + str(i+1) + '.edf')  # e.g., 'chb05_10.edf'
    
    # Specific seizure data is removed from certain sessions (indicated by their session number)
    if (i+1) == 6:  # Session 6: Remove seizure data between 417 and 532 seconds
        data = raw.get_data().T
        data = np.delete(data, range(417*256, 532*256), axis=0)  # 256 is the sampling rate
    elif (i+1) == 13:  # Session 13: Remove seizure data between 1086 and 1196 seconds
        data = raw.get_data().T
        data = np.delete(data, range(1086*256, 1196*256), axis=0)
    elif (i+1) == 16:  # Session 16: Remove seizure data between 2317 and 2413 seconds
        data = raw.get_data().T
        data = np.delete(data, range(2317*256, 2413*256), axis=0)
    elif (i+1) == 17:  # Session 17: Remove seizure data between 2451 and 2571 seconds
        data = raw.get_data().T
        data = np.delete(data, range(2451*256, 2571*256), axis=0)
    elif (i+1) == 22:  # Session 22: Remove seizure data between 2348 and 2465 seconds
        data = raw.get_data().T
        data = np.delete(data, range(2348*256, 2465*256), axis=0)
    else:
        data = raw.get_data().T  # For other sessions, simply get the data (no seizures)

    # Take the absolute value of the EEG data to focus on magnitude (removing negative values)
    data = np.absolute(data)
    
    # Block processing: Define the length of each block (m = 240 seconds)
    m = 240  # Block size of 240 seconds
    N = int(data.shape[0] / (256*m))  # Calculate the number of blocks (with 256 samples/second)

    # Loop through the EEG data and compute the maximum absolute value for each block
    for j in range(int(N)):
        index = np.arange(256*m*j, (256*m*(j+1)))  # Define indices for each block
        data_sub = np.max(data[index], 0)  # Get the maximum value of each channel within the block
        data_mat.append(data_sub)  # Append the maximum values to the data matrix

# Convert the list of block maxima into a NumPy array for further analysis
data_mat = np.array(data_mat)

# Plotting the entire matrix for all channels
fig, ax = plt.subplots()
ax.plot(data_mat)  # Plot block maxima for all channels
plt.show()  # Display the plot

# Save the processed data (block maxima) into a CSV file
header = ','.join(raw.ch_names)  # Create a header string with channel names
np.savetxt('data/eeg_wo_seizures.csv', data_mat, delimiter=',', header=header)  # Save the data with channel names

# Print the shape of the final matrix (number of blocks x number of channels)
print(data_mat.shape)
