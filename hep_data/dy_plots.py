import pandas as pd
import matplotlib.pyplot as plt

# list of CSV file paths
file_paths = ['./data/events_data_dy_Z_test.csv', './data/events_data_bkg_ZW.csv', './data/events_data_bkg_WW.csv', './data/events_data_bkg_ALL.csv', './data/events_data_bkg_TTBAR.csv']

# # read each CSV file into a DataFrame
# dfs = [pd.read_csv(file) for file in file_paths]


scaling_factors = [100.0, 1.0, 1.0, 1.0, 1.0]
colors = ['blue', 'green', 'orange', 'red', 'black']

import numpy as np
#num_bins = np.linspace(0, 1000, 51)
num_bins = np.linspace(-1, 1, 51)

# Read each CSV file into a DataFrame and plot stacked histogram with log y-axis and LaTeX labeling
plt.figure(figsize=(10, 6))
for i, file in enumerate(file_paths):
    # df = pd.read_csv(file)
    # plt.hist(df['reco_Z_masses'], bins=num_bins, histtype='step', stacked=True, alpha=0.5, color=colors[i], label=f'File {i+1}')

    df = pd.read_csv(file)
    counts, _ = np.histogram(df['cos_theta_pos'], bins=num_bins)
    scaled_counts = counts * scaling_factors[i]  # Scale the counts by the factor
    plt.hist(num_bins[:-1], num_bins, weights=scaled_counts, stacked=True, histtype='step', alpha=0.5, color=colors[i], label=f'File {i+1}')
    
plt.xlabel(r'$\text{mee}$')  # LaTeX labeling for x-axis
plt.ylabel(r'$\text{Events}$')  # LaTeX labeling for y-axis
plt.title('Stacked Histogram of Variable')
plt.yscale('log')  # Set y-axis to logarithmic scale
plt.legend()
plt.show()



