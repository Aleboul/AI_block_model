"""
Clustering Analysis of the EFAS Database in the European Domain

This script performs clustering analysis on the EFAS (European Flood Awareness System) database using an AI block model.
The analysis is based on extremal correlation methods and aims to identify significant patterns of extremal dependence 
within the dataset.

Dependencies:
- pandas: For data manipulation and analysis.
- numpy: For numerical computations and array manipulations.
- matplotlib: For plotting and visualizing results.
- seaborn: For enhanced data visualization aesthetics.
- geopandas: For geographic data manipulation and visualization.
- shapely: For geometric operations and representations.
- eco_alg: A custom module for clustering algorithms.

Key Steps:
1. **Data Preparation**: 
   - Load monthly data from CSV files and concatenate into a single DataFrame.
   - Clean the dataset by removing erroneous stations and handling missing values.
   - Convert the date column for time series analysis and group data into 14-day periods.

2. **Extremal Correlation Calculation**:
   - Calculate the extremal correlation matrix from the cleaned data.
   - Normalize the values and apply thresholds to identify significant relationships.

3. **Threshold Calibration**:
   - Calibrate thresholds according to SECO (Spatial Extremal Correlation Optimization) values, identifying two critical drops in the SECO metric.

4. **Clustering**:
   - Perform clustering on the extremal correlation matrix using a calibrated threshold value, which reveals distinct clusters with strong extremal dependence.

5. **Visualization**:
   - Visualize the extremal correlation matrix to highlight clusters.
   - Create a spatial representation of the clusters on a map, focusing on France, to analyze geographical patterns in extremal dependence.

The results of this analysis provide insights into the relationships and dependencies among different stations, facilitating better understanding of extreme events across the European domain.
"""


import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import sample

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import seaborn as sns
import geopandas as gpd
from shapely.geometry import Polygon
from matplotlib.patches import Rectangle

from itertools import islice
from shapely.ops import unary_union

import eco_alg

plt.style.use('qb-light.mplstyle')

# Load data from CSV files for each month and the station information

# Read the CSV files containing data for each month into separate DataFrames
january = pd.read_csv('data/january.csv')  # Load January data
february = pd.read_csv('data/february.csv')  # Load February data
# Load March data (note: "mars" likely means "March" in French)
mars = pd.read_csv('data/mars.csv')
april = pd.read_csv('data/april.csv')  # Load April data
may = pd.read_csv('data/may.csv')  # Load May data
june = pd.read_csv('data/june.csv')  # Load June data
july = pd.read_csv('data/july.csv')  # Load July data
# Load August data (note: "aout" is "August" in French)
august = pd.read_csv('data/aout.csv')
september = pd.read_csv('data/september.csv')  # Load September data
october = pd.read_csv('data/october.csv')  # Load October data
# Load November data (note: "novembre" is "November" in French)
november = pd.read_csv('data/novembre.csv')
december = pd.read_csv('data/december.csv')  # Load December data

# Read the stations information from a CSV file
stations = pd.read_csv('data/stations_w.csv')  # Load station metadata

# Concatenate the monthly DataFrames into a single DataFrame named 'data'
data = pd.concat([january, february, mars, april, may, june,
                 july, august, september, october, november, december])

# Convert the 'Historical_times' column to datetime format for time series analysis
data['Historical_times'] = pd.to_datetime(data['Historical_times'])

# Group the data by 14-day periods and compute the maximum value for each period
data = data.groupby(pd.Grouper(key="Historical_times", freq="14D")).max()

# Drop any rows that contain NaN values, which may have arisen from the aggregation
data = data.dropna()

# Print the shape of the resulting DataFrame to understand its dimensions (rows, columns)
print(data.shape)

# List of station IDs to be removed due to processing errors or irrelevant data (e.g., distance issues)
remove = ['4001', '4003', '4004', '4006', '4007', '4008',
          '4009', '4011', '4013', '4014', '4015', '4016',
          '1680', '1681', '1683', '1685', '1688', '1690',
          '2274', '2190', '2193', '2194', '2195']  # Exclude these stations from the dataset : remove Albania due to processing errors, all obtained time series are the same.

# Drop the specified columns (stations ID) from the DataFrame
data = data.drop(remove, axis=1)

# Get the number of columns (d) and rows (n) in the cleaned DataFrame
# Get the number of columns (features) after dropping specified stations
d = data.shape[1]
# Get the number of rows (observations) after removing NaN values
n = data.shape[0]

# Convert the cleaned DataFrame to a NumPy array for further processing
# Create a NumPy array from the DataFrame for numerical computations
matrix = data.to_numpy()

# Print the number of rows and columns in the final matrix to understand its dimensions
print(n, d)  # Display the shape of the NumPy array (number of observations and features)

# Compute the sampling version of the extremal correlation matrix
# Calculate the rank of each element and normalize by the number of observations
erank = np.array(data.rank() / (n + 1))
# Compute the outer maximum of ranks for extremal correlation
outer = (np.maximum(erank[:, :, None], erank[:, None, :])).sum(0) / n

# Calculate the extremal correlation coefficients
# Compute the extremal coefficients based on the outer matrix
extcoeff = -np.divide(outer, outer - 1)
# Threshold the coefficients to ensure they are above a minimum value
Theta = np.maximum(2 - extcoeff, 10e-5)

# Calibrate the threshold with the SECO (Spatial Extremal Correlation Optimization)
# Then plot

# Create an array of threshold values for calibration
_tau_ = np.array(np.arange(0.1, 0.3, step=0.0025))
value_SECO = []  # Initialize a list to store SECO values for each threshold

# Loop through each threshold value to compute SECO
for tau in _tau_:
    print(tau)  # Print the current threshold value being processed
    # Perform clustering on the extremal correlation matrix
    clusters = eco_alg.clust(Theta, n=1000, alpha=tau)
    # Compute the SECO value for the current clustering
    value = eco_alg.SECO(Theta, clusters)
    value_SECO.append(value)  # Store the computed SECO value

# Convert the list of SECO values to a NumPy array for further processing
value_SECO = np.array(value_SECO)
# Normalize SECO values for visualization
value_SECO = np.log(1 + value_SECO - np.min(value_SECO))
ind = np.argmin(value_SECO)  # Find the index of the minimum SECO value

# Plot the SECO values against the thresholds
fig, ax = plt.subplots()  # Create a new figure and axis for plotting
ax.plot(_tau_, value_SECO, marker='o', linestyle='solid',  # Plot the SECO values
        markerfacecolor='white', lw=1, markersize=2)
ax.set_ylabel('SECO')  # Label the y-axis
# Label the x-axis with a LaTeX formatted label
ax.set_xlabel(r'Treshold $\tau$')
plt.show()  # Display the plot
fig.savefig('results/seco_tau.pdf')  # Save the plot as a PDF file

# Use calibrated threshold for clustering
tau = 0.25  # Set a specific threshold value for clustering
# Perform clustering using the extremal correlation matrix and the defined threshold
O_hat = eco_alg.clust(Theta, alpha=tau, n=n)
# Compute SECO based on the clustering results
seco = eco_alg.SECO(Theta, O_hat)

index = []  # Initialize a list to store shuffled indices for clusters
for key, item in O_hat.items():  # Iterate over each cluster in the clustering results
    shuffled = sorted(item, key=lambda k: random.random()
                      )  # Shuffle the indices randomly
    index.append(shuffled)  # Append the shuffled indices to the list

# Flatten the list of indices into a single array
# Combine the lists into a single array for reordering
index = np.hstack(index)

# Create a new extremal correlation matrix based on the shuffled indices
# Reorder the extremal correlation matrix based on the shuffled indices
new_Theta = Theta[index, :][:, index]
# Initialize an array to hold the sizes of clusters
sizes = np.zeros(len(O_hat) + 1)

# Calculate sizes of each cluster
for key, item in O_hat.items():
    sizes[key] = len(item)  # Store the size of each cluster

# Calculate cumulative sizes for positioning rectangles in the plot
# Compute cumulative sizes of clusters for visualization
cusizes = np.cumsum(sizes) - 0.5

# Plot the emphasized block extremal correlation matrix
fig, ax = plt.subplots()  # Create a new figure and axis for plotting
# Display the extremal correlation matrix using a colormap
im = plt.imshow(new_Theta, cmap="Blues_r")
for i in range(0, len(O_hat)):  # Iterate over each cluster
    # Add rectangles to highlight clusters in the extremal correlation matrix
    ax.add_patch(Rectangle((cusizes[i], cusizes[i]), sizes[i + 1],
                 sizes[i + 1], edgecolor='#323232', fill=False, lw=2))
plt.colorbar(im)  # Add a color bar to the plot
ax.grid(False)  # Disable the grid on the plot
fig.savefig('results/clust_matrix.pdf')  # Save the plot as a PDF file

# Spatial representation of clusters
# Define a geographic area as a polygon
polys1 = gpd.GeoSeries(Polygon([(-16, 29), (43, 29), (43, 76), (-16, 76)]))
# Create a GeoDataFrame for the polygon with a specific coordinate reference system
df1 = gpd.GeoDataFrame({'geometry': polys1}).set_crs('epsg:4326')

# Define a color palette for cluster visualization
colors = ["steelblue", "lightblue", "darkorange", "chocolate", "darkseagreen", "limegreen", "darkslateblue",
          "royalblue", "yellow", "gold", "steelblue", "lightblue", "darkorange", "chocolate", "darkseagreen",
          "limegreen", "darkslateblue", "royalblue", "yellow", "gold"]

# Load world administrative boundaries for geographical context
# Read the world boundaries from a GeoJSON file
world = gpd.read_file("data/world-administrative-boundaries.geojson")
# Find the intersection of the world boundaries and the defined polygon
frankreich = gpd.overlay(world, df1, how='intersection')

# Sort clusters by size for consistent plotting
# Sort the clusters by their size in descending order
O_hat = dict(sorted(O_hat.items(), key=lambda i: -len(i[1])))
print(O_hat)  # Print the sorted clusters for reference
# Print the array representation of the sorted clusters for reference
print(np.array(O_hat))

# Set the style for the plot
# Use a seaborn color palette for qualitative data visualization
qualitative_colors = sns.color_palette("Paired", 12)
plt.style.use('seaborn-whitegrid')  # Set the plot style

# Create a new figure for the spatial representation of clusters
fig, ax = plt.subplots()
i = 0  # Initialize an index for color assignment
for clst in [1, 2, 3]:  # Iterate over the first three clusters for visualization
    # Select the stations belonging to the current cluster
    cluster = stations.iloc[O_hat[clst], :]
    y = cluster['StationLat']  # Get the latitude of the stations
    x = cluster['StationLon']  # Get the longitude of the stations
    # Combine latitude and longitude into coordinate pairs
    coordinate_ = np.c_[x, y]
    polygon = []  # Initialize a list to hold polygon geometries for visualization
    for coord in coordinate_:  # Iterate over the coordinates to create polygons
        polygon_geom = Polygon([((coord[0]-0.125), (coord[1]-0.125)),  # Define a square polygon around each coordinate
                               ((coord[0]+0.125), (coord[1]-0.125)),
                               ((coord[0]+0.125), (coord[1]+0.125)),
                               ((coord[0]-0.125), (coord[1]+0.125))])
        polygon.append(polygon_geom)  # Add the polygon to the list
        # Combine all polygons into a single geometry for plotting
        cu = unary_union(polygon)

    # Plot each polygon geometry on the map
    for geom in cu.geoms:
        xs, ys = geom.exterior.xy  # Get the x and y coordinates of the polygon boundary
        # Fill the polygon with color
        ax.fill(xs, ys, fc=qualitative_colors[i], ec='none')
        # Outline the polygon
        ax.plot(xs, ys, color=qualitative_colors[i], linewidth=0.75)
    i += 1  # Move to the next color for the next cluster

# Overlay the boundaries of France on the plot
# Draw the boundary of the intersection area
frankreich.boundary.plot(ax=ax, linewidth=0.5, color='black')
# Plot the intersection area with transparency
frankreich.plot(ax=ax, color='white', alpha=0.2)
ax.set_aspect(1.0)  # Set the aspect ratio of the plot to be equal
plt.show()  # Display the spatial representation of clusters
# Save the spatial plot as a PDF file
fig.savefig('results/clust_1_3_together.pdf')
