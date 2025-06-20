from datetime import datetime, timedelta
import os

import h5py
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def load_sea_ice_data(filepath):
    """Load sea ice data from HDF5 mat file"""
    return h5py.File(filepath, "r")


def matlab_datenum_to_datetime(datenum):
    """Convert MATLAB datenum to Python datetime"""
    # MATLAB datenum starts from January 1, 0000
    # Python datetime starts from January 1, 0001
    # MATLAB datenum 1 = January 1, 0001 in Python
    return (
        datetime.fromordinal(int(datenum))
        + timedelta(days=datenum % 1)
        - timedelta(days=366)
    )


def visualize_sea_ice_overview(data_file, save_plots=True):
    """Create overview visualization of all sea ice datasets"""

    with h5py.File(data_file, "r") as data:
        # Get dates
        dates_matlab = np.array(data["selected_dates"]).flatten()
        dates = [matlab_datenum_to_datetime(d) for d in dates_matlab]

        # Dataset names and their corresponding data
        datasets = {
            "Bremen ASI": "Bremen_ASI",
            "Bremen MODIS": "Bremen_MODIS",
            "NSIDC CDR": "NSIDC_CDR",
            "NSIDC NT2": "NSIDC_NT2",
            "OSI-401": "OSI401",
            "OSI-408": "OSI408",
        }

        # Coordinate mapping for each dataset
        coord_map = {
            "Bremen_ASI": ("x_asi", "y_asi"),
            "Bremen_MODIS": ("modis_x", "modis_y"),
            "NSIDC_CDR": ("x_cdr", "y_cdr"),
            "NSIDC_NT2": ("x_nt2", "y_nt2"),
            "OSI401": ("x_osi401", "y_osi401"),
            "OSI408": ("x_osi408", "y_osi408"),
        }

        # Get Bremen MODIS coordinates for reference rectangle
        modis_x = np.array(data["modis_x"]).flatten()
        modis_y = np.array(data["modis_y"]).flatten()
        modis_extent = (modis_x.min(), modis_x.max(), modis_y.min(), modis_y.max())

        # Calculate the correct aspect ratio based on MODIS coordinates
        modis_x_range = modis_x.max() - modis_x.min()
        modis_y_range = modis_y.max() - modis_y.min()
        aspect_ratio = modis_x_range / modis_y_range

        # Create subplot layout
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()

        for idx, (name, key) in enumerate(datasets.items()):
            ax = axes[idx]
            x_key, y_key = coord_map[key]
            
            # Get coordinates and data
            x_coords = np.array(data[x_key]).flatten()
            y_coords = np.array(data[y_key]).flatten()
            ice_data = np.array(data[key][0])  # First time step

            # Special handling for NSIDC NT2 y-coordinates
            if key == "NSIDC_NT2":
                y_coords = y_coords.reshape(-1)  # Ensure 1D array

            # Transpose data to match coordinate dimensions
            if ice_data.shape == (len(y_coords), len(x_coords)):
                # Data already in correct orientation (lat, lon)
                pass
            elif ice_data.shape == (len(x_coords), len(y_coords)):
                # Need to transpose to (lat, lon)
                ice_data = ice_data.T
            else:
                print(f"Warning: Shape mismatch for {name}: data {ice_data.shape}, "
                      f"x: {len(x_coords)}, y: {len(y_coords)}")

            # Create visualization with real coordinates
            # Use masked array to handle NaN values separately
            masked_data = np.ma.masked_invalid(ice_data)
            
            # Create colormap: Blues for ice concentration, gray for land (NaN)
            cmap = plt.cm.Blues.copy()
            cmap.set_bad('gray', 1.0)  # Set color for NaN values
            
            mesh = ax.pcolormesh(x_coords, y_coords, masked_data, 
                                cmap=cmap, shading='auto', vmin=0, vmax=1)
            ax.set_title(f"{name}\nShape: {ice_data.shape}", fontsize=12)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")

            # Set consistent aspect ratio for all plots
            ax.set_aspect(aspect_ratio)
            
            # Invert y-axis for Bremen MODIS
            if key == "Bremen_MODIS":
                ax.invert_yaxis()

            # Add colorbar with clear labels
            cbar = plt.colorbar(mesh, ax=ax, shrink=0.8)
            cbar.set_label('Sea Ice Concentration')
            cbar.set_ticks([0, 0.5, 1])
            cbar.set_ticklabels(['0 (open water)', '0.5', '1 (full ice)'])
            
            # Add MODIS extent rectangle to other datasets
            if key != "Bremen_MODIS":
                min_x, max_x, min_y, max_y = modis_extent
                rect = plt.Rectangle(
                    (min_x, min_y), 
                    max_x - min_x, 
                    max_y - min_y,
                    fill=False, 
                    edgecolor='red', 
                    linewidth=2
                )
                ax.add_patch(rect)

        plt.tight_layout()
        plt.suptitle(
            "Arctic Sea Ice Datasets Overview (First Time Step)", fontsize=16, y=1.02
        )

        if save_plots:
            plt.savefig("sea_ice_overview.png", dpi=300, bbox_inches="tight")
        plt.close()  # 关闭图形释放内存


def visualize_timestep(modis_x, modis_y, inputs, label, timestep, output_dir):
    """
    Visualize one timestep of preprocessed data and save to file.
    
    Args:
        modis_x: 1D array of x coordinates
        modis_y: 1D array of y coordinates
        inputs: List of 5 input arrays (each 2D)
        label: 2D label array
        timestep: Current timestep index
        output_dir: Output directory for images
    """
    # Names for the input channels
    input_names = ["Bremen_ASI", "NSIDC_CDR", "NSIDC_NT2", "OSI401", "OSI408"]
    
    # Create figure with 6 subplots (5 inputs + 1 label)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Compute aspect ratio
    x_min, x_max = modis_x.min(), modis_x.max()
    y_min, y_max = modis_y.min(), modis_y.max()
    aspect_ratio = (x_max - x_min) / (y_max - y_min)
    
    # Plot each input channel
    for i in range(5):
        ax = axes[i]
        data = inputs[i]
        masked_data = np.ma.masked_invalid(data)
        cmap = plt.cm.Blues.copy()
        cmap.set_bad('gray', 1.0)
        mesh = ax.pcolormesh(modis_x, modis_y, masked_data, 
                            cmap=cmap, shading='auto', vmin=0, vmax=1)
        ax.set_title(input_names[i])
        ax.set_aspect(aspect_ratio)
        ax.invert_yaxis()
        plt.colorbar(mesh, ax=ax, shrink=0.6)
    
    # Plot the label
    ax = axes[5]
    masked_label = np.ma.masked_invalid(label)
    mesh = ax.pcolormesh(modis_x, modis_y, masked_label, 
                        cmap=cmap, shading='auto', vmin=0, vmax=1)
    ax.set_title("Bremen_MODIS (Label)")
    ax.set_aspect(aspect_ratio)
    ax.invert_yaxis()
    plt.colorbar(mesh, ax=ax, shrink=0.6)
    
    # Set main title and layout
    fig.suptitle(f"Timestep {timestep}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"timestep_{timestep:04d}.png"), 
                dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    """Main function to run all visualizations"""
    data_file = "assets/sea_ice_dataset_withoutint.mat"

    print("Creating sea ice data visualizations...")

    # Overview of all datasets
    print("1. Creating overview plot...")
    visualize_sea_ice_overview(data_file)

    print("Visualizations complete!")


if __name__ == "__main__":
    main()