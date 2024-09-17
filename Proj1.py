import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# File paths for the three .txt datasets
file_paths = ['groupA.txt', 'groupB.txt', 'groupC.txt']

# Function to read, normalize, and plot data with a separation line
def process_and_plot(file_path):
    # Step 1: Read the .txt file
    df = pd.read_csv(file_path, sep=',', header=None, names=['Cost', 'Weight', 'Type'])
    
    # Step 2: Separate features (Cost, Weight) and target (Type)
    X = df[['Cost', 'Weight']].astype(float)  # Ensure the data is in float
    y = df['Type'].astype(int)  # Ensure the target (Type) is integer

    # Step 3: Normalize the data (Min-Max Scaling)
    min_max_scaler = MinMaxScaler()
    X_min_max_scaled = min_max_scaler.fit_transform(X)

    # Step 4: Plot the original and normalized data
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Original Data Plot
    ax[0].scatter(df['Cost'], df['Weight'], c=df['Type'], cmap='bwr', alpha=0.5)
    ax[0].set_title(f'Original Data - {file_path}')
    ax[0].set_xlabel('Cost')
    ax[0].set_ylabel('Weight')

    # Min-Max Scaled Data Plot
    ax[1].scatter(X_min_max_scaled[:, 0], X_min_max_scaled[:, 1], c=y, cmap='bwr', alpha=0.5)
    ax[1].set_title(f'Min-Max Normalized Data - {file_path}')
    ax[1].set_xlabel('Cost (Normalized)')
    ax[1].set_ylabel('Weight (Normalized)')
    
    # Set axis limits to [0, 1] for Min-Max Normalized plot
    ax[1].set_xlim(0, 1)
    ax[1].set_ylim(0, 1)

    # Step 5: Add a manually estimated separation line (example line)
    # Example: Let's assume the points (0.2, 0.3) and (0.8, 0.7) form the boundary
    x_vals = np.array([0, 1])  # x values range from 0 to 1
    m = (0.7 - 0.3) / (0.8 - 0.2)  # Calculate slope between two points
    b = 0.3 - m * 0.2  # Calculate intercept using one point

    # Equation of the line: y = mx + b
    y_vals = m * x_vals + b

    # Plot the separation line
    ax[1].plot(x_vals, y_vals, '--', color='green', label='Separation Line')
    ax[1].legend()

    # Display the plots
    plt.tight_layout()
    plt.show()

# Process and plot each dataset
for file_path in file_paths:
    process_and_plot(file_path)
