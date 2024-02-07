import pandas as pd
import matplotlib.pyplot as plt

# list of CSV file paths
file_paths = ['./data/events_data_bkg_ZW.csv', './data/events_data_bkg_WW.csv', './data/events_data_bkg_ALL.csv', './data/events_data_bkg_TTBAR.csv', './data/events_data_dy_Z.csv']

# # read each CSV file into a DataFrame
# dfs = [pd.read_csv(file) for file in file_paths]



colors = ['blue', 'green', 'orange', 'red', 'black']

# Read each CSV file into a DataFrame and plot stacked histogram with log y-axis and LaTeX labeling
plt.figure(figsize=(10, 6))
for i, file in enumerate(file_paths):
    df = pd.read_csv(file)
    plt.hist(df['reco_Z_masses'], bins=100, histtype='step', stacked=True, alpha=0.5, color=colors[i], label=f'File {i+1}')
    
plt.xlabel(r'$\text{mee}$')  # LaTeX labeling for x-axis
plt.ylabel(r'$\text{Events}$')  # LaTeX labeling for y-axis
plt.title('Stacked Histogram of Variable')
plt.yscale('log')  # Set y-axis to logarithmic scale
plt.legend()
plt.show()
