import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import os
import requests
from PIL import Image
from io import BytesIO
import matplotlib.animation as animation
import warnings
import matplotlib
import streamlit as st
import subprocess


warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

# FILEPATH = "A22_Step1_pixel_F1_0_45053.xlsx"
# MEASUREMENT_FREQUENCY_HZ = 50  # Samples per second
# BREATH_CYCLE_SECONDS = 2  # Seconds per full breath cycle
# RECORDING_DURATION_MINUTES = 15  # Total data duration in minutes

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run data pipeline.")
    parser.add_argument("filepath", type=str, help="Path to the input Excel file")
    parser.add_argument("--freq", type=int, default=50, help="Measurement frequency in Hz (default: 50)")
    parser.add_argument("--cycle", type=int, default=2, help="Breath cycle duration in seconds (default: 2)")
    parser.add_argument("--duration", type=int, default=15, help="Recording duration in minutes (default: 15)")
    return parser.parse_args()

args = parse_arguments()

FILEPATH = args.filepath
MEASUREMENT_FREQUENCY_HZ = args.freq
BREATH_CYCLE_SECONDS = args.cycle
RECORDING_DURATION_MINUTES = args.duration

# Define the local file path
file_path = os.path.expanduser(FILEPATH)

print("Trying to load the dataset...")

# Load the dataset
try:
    df = pd.read_excel(file_path, engine='openpyxl')
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    exit()

# Ensure Timestamp is numeric
df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
# Drop rows with invalid timestamps (if any)
df = df.dropna(subset=["Timestamp"])

# Create a global_df with only the first two columns
global_df = df.iloc[:, :2]  # Keep only the first two columns
# Ensure Timestamp is numeric
global_df["Timestamp"] = pd.to_numeric(global_df["Timestamp"], errors="coerce")
# Drop rows with invalid timestamps (if any)
global_df = global_df.dropna(subset=["Timestamp"])

# img_url = 'https://drive.google.com/uc?id=18bC43PP_jEsOoV866syDD8YdrbSp_6hH'
# try:
#     response = requests.get(img_url, stream=True)
#     response.raise_for_status()  # Check for HTTP errors

#     # Read the image
#     img = Image.open(BytesIO(response.content))

#     # Open in a new pop-up window
#     img.show(title="Dataset Loading...")
# except Exception as e:
#     print(f"Warning: Failed to load image ({e}). Continuing execution...")

print("Dataset loaded successfully.")

# Initialize an empty list to store the timestamps of the maxima
max_timestamps = []
max_values = []

# Loop through 100-time-step intervals
start = 0
end = int(global_df['Timestamp'].max())  # Find the maximum timestamp

interval_start = 0
interval_end = int(MEASUREMENT_FREQUENCY_HZ * BREATH_CYCLE_SECONDS)

while interval_end <= end:
    interval_data = global_df[
        (global_df['Timestamp'] >= interval_start) & (global_df['Timestamp'] < interval_end)
        ]
    if not interval_data.empty:
        max_row = interval_data.loc[interval_data['Global_impedance'].idxmax()]
        max_timestamps.append(max_row['Timestamp'])
        max_values.append(max_row['Global_impedance'])
        interval_start = max_row['Timestamp'] + int(MEASUREMENT_FREQUENCY_HZ * BREATH_CYCLE_SECONDS * 0.5)
        interval_end = interval_start + int(MEASUREMENT_FREQUENCY_HZ * BREATH_CYCLE_SECONDS * 1)
    else:
        break
# Print results
print(f"Detected {len(max_values)} peaks.")

# Initialize lists to store minima timestamps and values
min_timestamps = []
min_values = []

# Find the first minimum between time 40 and the first maximum
first_min_interval = global_df[(global_df['Timestamp'] >= 0) & (global_df['Timestamp'] < max_timestamps[0])]
if not first_min_interval.empty:
    first_min_row = first_min_interval.loc[first_min_interval['Global_impedance'].idxmin()]
    min_timestamps.append(first_min_row['Timestamp'])
    min_values.append(first_min_row['Global_impedance'])

# Find subsequent minima between each pair of consecutive maxima
for i in range(len(max_timestamps) - 1):
    min_interval = global_df[
        (global_df['Timestamp'] >= max_timestamps[i]) & (global_df['Timestamp'] < max_timestamps[i + 1])
        ]
    if not min_interval.empty:
        min_row = min_interval.loc[min_interval['Global_impedance'].idxmin()]
        min_timestamps.append(min_row['Timestamp'])
        min_values.append(min_row['Global_impedance'])
print(f"Detected {len(min_values)} troughs. (Should be identical as number of peaks)")

# Extract the base name without the extension
base_name = os.path.splitext(FILEPATH)[0]
# Create the full path for the results folder, inside the 'results' directory
results_dir = os.path.join("results", base_name)
# Create the directory if it doesn't exist
os.makedirs(results_dir, exist_ok=True)
print(f"Results directory created: {results_dir}")

# Define the output file path
plot_path = os.path.join(results_dir, "impedance_plot.png")
# Create and save the plot
plt.figure(figsize=(100, 6))  # Adjust figure size for local display
plt.plot(global_df['Timestamp'], global_df['Global_impedance'], label='Global Impedance', color='blue')

# Highlight the maxima
plt.scatter(max_timestamps, max_values, color='red', label='Maxima', zorder=5)

# Highlight the minima
plt.scatter(min_timestamps, min_values, color='green', label='Minima', zorder=5)

# Add labels and title
plt.xlabel('Timestep')
plt.ylabel('Global Impedance')
plt.title('Global Impedance Over Time with Highlighted Maxima and Minima')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

# Save the plot instead of displaying
plt.savefig(plot_path, dpi=300)  # High resolution for better clarity
plt.close()  # Close the plot to free memory

print(f"Plot saved successfully at: {plot_path}")

# Calculate the number of full 5-minute windows
num_full_windows = RECORDING_DURATION_MINUTES // 5

# Initialize lists for each zone
zones = [[] for _ in range(num_full_windows)]

# Iterate through the minima and maxima
for i in range(len(min_timestamps)):
    # Determine which zone the peak belongs to based on the max timestamp
    for j in range(num_full_windows):
        if min_timestamps[i] <= (j + 1) * 5 * 60 * MEASUREMENT_FREQUENCY_HZ:
            zones[j].append((min_values[i], max_values[i]))
            break

# Print the results for each zone
for zone_idx, zone in enumerate(zones, 1):
    print(f"Zone {zone_idx}: {len(zone)} cycles")

# Path for the global analysis file
analysis_file_path = os.path.join(results_dir, "global_analysis.txt")

# Open the file to write (will overwrite if file exists)
with open(analysis_file_path, 'w') as f:
    # Loop over each zone and compute the analysis
    for zone_idx, zone in enumerate(zones, 1):
        # Convert the zone to numpy arrays
        EILI = np.array([a for a, b in zone])
        EELI = np.array([b for a, b in zone])
        delta_z = np.array([b - a for a, b in zone])

        # Calculate lung strain element-wise
        lung_strain = EELI / delta_z

        # Calculate averages for the zone
        avg_EILI = np.mean(EILI)
        avg_EELI = np.mean(EELI)
        avg_delta_z = np.mean(delta_z)
        avg_lung_strain = np.mean(lung_strain)

        # Print to console
        print(f"Zone {zone_idx}:")
        print(f"EILI: {avg_EILI}")
        print(f"EELI: {avg_EELI}")
        print(f"Delta Z: {avg_delta_z}")
        print(f"Lung Strain: {avg_lung_strain}")
        print()  # Empty line for separation

        # Write to the file
        f.write(f"Zone {zone_idx}:\n")
        f.write(f"EILI: {avg_EILI}\n")
        f.write(f"EELI: {avg_EELI}\n")
        f.write(f"Delta Z: {avg_delta_z}\n")
        f.write(f"Lung Strain: {avg_lung_strain}\n")
        f.write("\n")  # Empty line for separation

print(f"Analysis written to {analysis_file_path}")

# ****************************************************************************
# Begin pixel analysis

# Convert min_timestamps and max_timestamps to a set for faster lookup
timestamp_set = set(min_timestamps + max_timestamps)
# Filter the df to only include rows with these timestamps
filtered_df = df[df['Timestamp'].isin(timestamp_set)]
# Keep only the first 1026 columns (assuming the first column is 'Timestamp' and you need the next 1025 columns)
filtered_df = filtered_df.iloc[:, :1026]
# Display the resulting filtered DataFrame
print(filtered_df.head())

# Extract only pixel columns (Column1 to Column1024)
pixel_columns = [f"Column{i}" for i in range(1, 1025)]

# Get pixel data from the dataframe
pixel_data = filtered_df[pixel_columns].iloc[0:].reset_index(drop=True)  # Skip the header row

# Initialize a dictionary to store the results
results = {}

# Loop through the data in pairs (min, max, min... pattern)
for pair_idx in range(0, len(pixel_data) - 1, 2):  # Ensure there's always a next row to pair
    row1 = pixel_data.iloc[pair_idx].astype(float)  # Inhaled point
    row2 = pixel_data.iloc[pair_idx + 1].astype(float)  # Exhaled point

    # Calculate absolute difference between inhaled and exhaled points
    differences = row2 - row1

    # Find the maximum difference and calculate 9% threshold
    max_diff = differences.max()
    threshold = 0.09 * max_diff

    # Identify pixels exceeding the threshold
    exceeding_pixels = differences[differences > threshold].index.tolist()

    # Store the result for the pair (breath cycle)
    results[pair_idx // 2] = exceeding_pixels

# Optionally, print the results to the console
# Display the results
for pair, pixels in results.items():
    print(f"Breath {pair + 1}: {len(pixels)} pixels are ventilated.")

# Initialize a dictionary to store active pixel counts for each zone
pixel_counts_zones = {i: {f"Column{i}": 0 for i in range(1, 1025)} for i in range(len(zones))}

# Loop through all the pairs and track active pixels for each zone
for pair_idx in range(0, len(pixel_data), 2):  # Step by 2 (each pair is consecutive rows)
    # Determine the zone based on pair index
    index = 0
    curr = -1
    for i in range(len(zones)):
        if curr + len(zones[i]) < pair_idx // 2:
            index += 1
            curr += len(zones[i])
        else:
            break
    pixel_counts = pixel_counts_zones[index]

    # Extract inhaled and exhaled points
    row1 = pixel_data.iloc[pair_idx].astype(float)
    row2 = pixel_data.iloc[pair_idx + 1].astype(float)

    # Normalize to the range [0, 1]
    max_value = max(row1.max(), row2.max())
    row1_normalized = row1 / max_value
    row2_normalized = row2 / max_value

    # Get indices of active pixels from previous results
    active_pixel_indices = results[pair_idx // 2]

    # Convert active_pixel_indices to numeric values (zero-based)
    numeric_indices = [
        int(col_name.replace("Column", "")) - 1  # Convert 'ColumnX' to index X-1
        for col_name in active_pixel_indices
    ]

    # Increment the active pixel counts for each active pixel in the current zone
    for idx in numeric_indices:
        column_name = f"Column{idx + 1}"  # Convert back to 1-based column name
        pixel_counts[column_name] += 1
# Create activation images for each zone
activation_images = {i: np.zeros((32, 32)) for i in range(len(zones))}

# Populate activation images with frequency data
for zone, pixel_counts in pixel_counts_zones.items():
    for idx, count in pixel_counts.items():
        if count > 0:  # If the pixel was activated at least once
            column_idx = int(idx.replace("Column", "")) - 1
            row, col = np.unravel_index(column_idx, (32, 32))
            activation_images[zone][row, col] = count

# Plot activation maps for each zone
for zone, activation_image in activation_images.items():
    plot_path = os.path.join(results_dir, f"Zone {zone + 1} Ventilated Area.png")
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.imshow(activation_image, cmap='Reds', interpolation='nearest')
    fig.colorbar(cax, ax=ax, label='Frequency of Activation')
    ax.set_title(f"Zone {zone + 1}: Pixel Activation Frequencies")
    ax.axis('off')
    plt.savefig(plot_path, dpi=300)
    plt.close()


# Step 2: Create the summary Excel file for all zones
summary_file_path = os.path.join(results_dir, 'active_pixel_summary.xlsx')

with pd.ExcelWriter(summary_file_path) as writer:
    for zone, pixel_counts in pixel_counts_zones.items():
        summary_data = {
            "Pixel": [f"Column{i}" for i in range(1, 1025)],
            "Activation Count": [pixel_counts[f"Column{i}"] for i in range(1, 1025)],
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name=f'Zone {zone + 1}', index=False)

print(f"Summary saved to: {summary_file_path}")


pixel_analysis_file_path = os.path.join(results_dir, "pixel_analysis_all_breath_cycles.txt")
pixel_analysis_summary_file_path = os.path.join(results_dir, "pixel_analysis_summary.txt")

mean_strain_summary = []
lsv_summary = []
gi_summary = []

with open(pixel_analysis_file_path, 'w') as f:
    for pair_idx in range(0, len(pixel_data) - 1, 2):
        row1 = pixel_data.iloc[pair_idx].astype(float)
        row2 = pixel_data.iloc[pair_idx + 1].astype(float)
        z = row2 - row1
        strain = z / row1

        ventilated_pixels = results[pair_idx // 2]
        ventilated_pixel_strain = strain[ventilated_pixels].values
        mean_strain = np.mean(ventilated_pixel_strain)
        median_strain = np.median(ventilated_pixel_strain)

        lung_strain_variation = np.sum(np.abs(ventilated_pixel_strain - mean_strain)) / len(ventilated_pixel_strain)
        gi = np.sum(np.abs(ventilated_pixel_strain - median_strain)) / sum(ventilated_pixel_strain)
        mean_strain_summary.append(mean_strain)
        lsv_summary.append(lung_strain_variation)
        gi_summary.append(gi)

        print(f"Breath {pair_idx // 2 + 1}: Mean Strain = {mean_strain}, Heterogeneity Index = {gi}, Lung Strain Variation = {lung_strain_variation}")
        f.write(f"Breath {pair_idx // 2 + 1}: Mean Strain = {mean_strain}, Heterogeneity Index = {gi}, Lung Strain Variation = {lung_strain_variation}\n")

print(f"Pixel analysis written to {pixel_analysis_file_path}")

with open(pixel_analysis_summary_file_path, 'w') as f:
    num_zones = len(zones)
    curr = 0
    for i in range(len(zones)):
        num_breath = len(zones[i])
        strains = mean_strain_summary[curr:curr + num_breath]
        lsvs = lsv_summary[curr:curr + num_breath]
        gis = gi_summary[curr:curr + num_breath]
        curr += num_breath
        strain_avg = np.mean(strains)
        lsv_avg = np.mean(lsvs)
        gi_avg = np.mean(gis)
        print(f"Zone {i + 1}. number of breath: {num_breath}")
        print(f"Mean Strain Average = {strain_avg}")
        print(f"Mean LSV Average = {lsv_avg}")
        print(f"Mean GI Average = {gi_avg}")

        f.write(f"Zone {i + 1}. number of breath: {num_breath}\n")
        f.write(f"Mean Strain Average = {strain_avg}\n")
        f.write(f"Mean LSV Average = {lsv_avg}\n")
        f.write(f"Mean GI Average = {gi_avg}\n")
        f.write("\n")

print(f"Pixel analysis summary written to {pixel_analysis_summary_file_path}")


num_frames = len(pixel_data) // 2  # Number of breath cycles
frame_duration = 0.05  # Time per frame in seconds (~22.5 sec total for 450 frames)
video_filename = os.path.join(results_dir, "lung_strain_evolution.mp4")

# Compute global min and max strain for consistent color scale
min_strain, max_strain = 0, 0
for pair_idx in range(0, len(pixel_data) - 1, 2):
    row1 = pixel_data.iloc[pair_idx].astype(float)
    row2 = pixel_data.iloc[pair_idx + 1].astype(float)
    z = row2 - row1
    strain = z / row1
    ventilated_pixels = results[pair_idx // 2]
    ventilated_pixel_strain = strain[ventilated_pixels].values

    max_strain = max(max_strain, max(ventilated_pixel_strain))
    min_strain = min(min_strain, min(ventilated_pixel_strain))

# Create figure and axis
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(np.zeros((32, 32)), cmap='hot', interpolation='nearest', vmin=min_strain, vmax=max_strain)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Strain Value')

# Fix aspect ratio
ax.set_aspect('equal')

def update(frame_idx):
    row1 = pixel_data.iloc[2 * frame_idx].astype(float)
    row2 = pixel_data.iloc[2 * frame_idx + 1].astype(float)
    z = row2 - row1
    strain = z / row1
    ventilated_pixels = results[frame_idx]

    # Create a 32x32 strain grid
    strain_grid = np.zeros((32, 32))

    for idx in ventilated_pixels:
        pixel_index = int(idx.replace('Column', '')) - 1
        row, col = np.unravel_index(pixel_index, (32, 32))
        strain_grid[row, col] = strain[pixel_index]

    # Update the heatmap (reuse the same image object)
    im.set_data(strain_grid)

    ax.set_title(f"Lung Strain for Breath {frame_idx + 1}")
    return im,

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=num_frames, blit=False)

# Save as MP4 video
writer = animation.FFMpegWriter(fps=int(1 / frame_duration))
ani.save(video_filename, writer=writer)
plt.close()

print(f"Video saved at {video_filename}")
