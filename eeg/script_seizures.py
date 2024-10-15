"""
EEG Block Maxima Computation for CHB-MIT Database

This script processes EEG data from the CHB-MIT Scalp EEG Database to compute block maxima 
for multiple seizure events. The signals are sampled at 256 Hz, and most recordings contain 
23 channels, following the 10-20 electrode system. 

The process involves:
1. Loading multiple EEG recordings from the database.
2. Extracting specific seizure event windows based on pre-defined time intervals.
3. Concatenating the EEG data from these seizures.
4. Downsampling the data by computing the maximum absolute value within 5-second blocks (variable 'm').
5. Saving the block maxima as a time-series in a CSV file for further analysis.

Outputs:
- A CSV file ('eeg_seizures.csv') containing the block maxima for all channels.
- A plot showing the time series of block maxima.

Dependencies:
- mne: for reading EEG data in EDF format.
- matplotlib: for data visualization.
- numpy: for numerical computations.
"""

# Import necessary libraries for EEG data processing and visualization
import mne  # MNE for reading and processing EEG/MEG data
import matplotlib.pyplot as plt  # For plotting
import numpy as np  # For numerical computations

# Load EEG data from the CHB-MIT Scalp EEG database (Patient 05, session 06)
raw = mne.io.read_raw_edf('data/chb05_06.edf')  # Load the EEG data from the .edf file
data = raw.get_data().T  # Get EEG data and transpose it so that rows represent time points and columns represent channels
# Extract a segment of the EEG data corresponding to a seizure event (between 417 and 532 seconds)
data_sei_06 = data[(417*256):(532*256),:]  # 256 is the sampling rate (samples per second)
print(data_sei_06)  # Print the extracted seizure data

# Initialize an empty list to store EEG data from multiple seizure events
data_mat = []

# Loop through each row (time point) of the extracted seizure data and append it to the data matrix
for i in range(0, data_sei_06.shape[0]):
    data_mat.append(data_sei_06[i,:])

# Print the shape of the data matrix (time points x channels)
print(np.array(data_mat).shape)

# Repeat the same process for seizure data from different sessions (Patient 05, session 13)
raw = mne.io.read_raw_edf('data/chb05_13.edf')  # Load EEG data from session 13
data = raw.get_data().T  # Transpose the data matrix
# Extract another seizure event (between 1086 and 1196 seconds)
data_sei_06 = data[(1086*256):(1196*256),:]  # 256 samples per second, seizure event window

# Append the seizure data from session 13 to the main data matrix
for i in range(0, data_sei_06.shape[0]):
    data_mat.append(data_sei_06[i,:])

# Print the shape of the updated data matrix
print(np.array(data_mat).shape)

# Repeat the process for seizure data from session 16
raw = mne.io.read_raw_edf('data/chb05_16.edf')  # Load EEG data from session 16
data = raw.get_data().T  # Transpose the data matrix
# Extract a seizure event (between 2317 and 2413 seconds)
data_sei_06 = data[(2317*256):(2413*256),:]

# Append the seizure data from session 16 to the main data matrix
for i in range(0, data_sei_06.shape[0]):
    data_mat.append(data_sei_06[i,:])

# Print the shape of the data matrix again
print(np.array(data_mat).shape)

# Repeat the process for seizure data from session 17
raw = mne.io.read_raw_edf('data/chb05_17.edf')  # Load EEG data from session 17
data = raw.get_data().T  # Transpose the data matrix
# Extract a seizure event (between 2451 and 2571 seconds)
data_sei_06 = data[(2451*256):(2571*256),:]

# Append the seizure data from session 17 to the main data matrix
for i in range(0, data_sei_06.shape[0]):
    data_mat.append(data_sei_06[i,:])

# Print the updated shape of the data matrix
print(np.array(data_mat).shape)

# Repeat the process for seizure data from session 22
raw = mne.io.read_raw_edf('data/chb05_22.edf')  # Load EEG data from session 22
data = raw.get_data().T  # Transpose the data matrix
# Extract a seizure event (between 2348 and 2465 seconds)
data_sei_06 = data[(2348*256):(2465*256),:]

# Append the seizure data from session 22 to the main data matrix
for i in range(0, data_sei_06.shape[0]):
    data_mat.append(data_sei_06[i,:])

# Convert the list of seizure data into a numpy array for easier manipulation
data = np.array(data_mat)
# Print the shape of the final combined data matrix (time points x channels)
print(np.array(data_mat).shape)

# Downsampling the data for analysis
m = 5  # Define the window size for downsampling (5 seconds)
N = int(data.shape[0] / (256*m))  # Calculate how many 5-second windows we have

# Initialize a new matrix to store the downsampled data
data_mat = np.zeros([N,23])  # 23 channels, N windows

# Loop through the EEG data and downsample it by taking the maximum absolute value within each window
for j in range(int(N)):
        index = np.arange(256*m*j, (256*m*(j+1)))  # Define the indices for the current window
        data_sub = np.max(np.absolute(data[index]), 0)  # Get the maximum absolute value for each channel
        data_mat[j] = data_sub  # Store the downsampled values in the matrix

# Print the final downsampled matrix
print(data_mat)

# Plot the downsampled seizure data
fig, ax = plt.subplots()  # Create a new figure and axis
ax.plot(data_mat)  # Plot the data
plt.show()  # Display the plot

# Save the downsampled data as a CSV file
header = ','.join(raw.ch_names)  # Create a header from the channel names
np.savetxt('data/eeg_seizures.csv', data_mat, delimiter = ',', header=header)  # Save the data as 'eeg_seizures.csv'
