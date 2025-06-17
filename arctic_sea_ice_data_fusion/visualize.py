import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.dates as mdates
from datetime import datetime, timedelta

def load_sea_ice_data(filepath):
    """Load sea ice data from HDF5 mat file"""
    return h5py.File(filepath, 'r')

def matlab_datenum_to_datetime(datenum):
    """Convert MATLAB datenum to Python datetime"""
    # MATLAB datenum starts from January 1, 0000
    # Python datetime starts from January 1, 0001
    # MATLAB datenum 1 = January 1, 0001 in Python
    return datetime.fromordinal(int(datenum)) + timedelta(days=datenum%1) - timedelta(days=366)

def visualize_sea_ice_overview(data_file, save_plots=True):
    """Create overview visualization of all sea ice datasets"""
    
    with h5py.File(data_file, 'r') as data:
        # Get dates
        dates_matlab = np.array(data['selected_dates']).flatten()
        dates = [matlab_datenum_to_datetime(d) for d in dates_matlab]
        
        # Dataset names and their corresponding data
        datasets = {
            'Bremen ASI': 'Bremen_ASI',
            'Bremen MODIS': 'Bremen_MODIS', 
            'NSIDC CDR': 'NSIDC_CDR',
            'NSIDC NT2': 'NSIDC_NT2',
            'OSI-401': 'OSI401',
            'OSI-408': 'OSI408'
        }
        
        # Create subplot layout
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        for idx, (name, key) in enumerate(datasets.items()):
            ax = axes[idx]
            
            # Get first time step for visualization
            ice_data = np.array(data[key][0])  # First time step
            
            # Handle NaN values for visualization
            ice_data_vis = np.where(np.isnan(ice_data), 0, ice_data)
            
            # Create visualization
            im = ax.imshow(ice_data_vis, cmap='Blues', origin='lower')
            ax.set_title(f'{name}\nShape: {ice_data.shape}', fontsize=12)
            ax.set_xlabel('Longitude grid')
            ax.set_ylabel('Latitude grid')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, shrink=0.8)
        
        plt.tight_layout()
        plt.suptitle('Arctic Sea Ice Datasets Overview (First Time Step)', fontsize=16, y=1.02)
        
        if save_plots:
            plt.savefig('sea_ice_overview.png', dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图形释放内存

def visualize_time_series(data_file, save_plots=True):
    """Visualize time series of sea ice extent/area"""
    
    with h5py.File(data_file, 'r') as data:
        dates_matlab = np.array(data['selected_dates']).flatten()
        dates = [matlab_datenum_to_datetime(d) for d in dates_matlab]
        
        datasets = {
            'Bremen ASI': 'Bremen_ASI',
            'Bremen MODIS': 'Bremen_MODIS',
            'NSIDC CDR': 'NSIDC_CDR', 
            'NSIDC NT2': 'NSIDC_NT2',
            'OSI-401': 'OSI401',
            'OSI-408': 'OSI408'
        }
        
        plt.figure(figsize=(15, 10))
        
        for name, key in datasets.items():
            ice_data = np.array(data[key])
            
            # Calculate valid (non-NaN) pixel count for each time step
            valid_pixels = []
            for t in range(ice_data.shape[0]):
                valid_count = np.sum(~np.isnan(ice_data[t]))
                valid_pixels.append(valid_count)
            
            plt.plot(dates, valid_pixels, label=name, marker='o', markersize=3)
        
        plt.xlabel('Date')
        plt.ylabel('Valid Pixels Count')
        plt.title('Sea Ice Data Coverage Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Format x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('sea_ice_time_series.png', dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图形释放内存

def visualize_dataset_comparison(data_file, time_step=0, save_plots=True):
    """Compare different datasets at a specific time step"""
    
    with h5py.File(data_file, 'r') as data:
        dates_matlab = np.array(data['selected_dates']).flatten()
        date = matlab_datenum_to_datetime(dates_matlab[time_step])
        
        datasets = {
            'Bremen ASI': 'Bremen_ASI',
            'NSIDC CDR': 'NSIDC_CDR',
            'NSIDC NT2': 'NSIDC_NT2', 
            'OSI-401': 'OSI401',
            'OSI-408': 'OSI408',
            'Bremen MODIS': 'Bremen_MODIS',
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, (name, key) in enumerate(datasets.items()):
            ax = axes[idx]
            ice_data = np.array(data[key][time_step])
            
            # Create binary ice/no-ice visualization
            ice_binary = np.where(np.isnan(ice_data), 0, 1)
            
            im = ax.imshow(ice_binary, cmap='Blues', origin='lower')
            ax.set_title(f'{name}\n{date.strftime("%Y-%m-%d")}')
            ax.set_xlabel('Longitude grid')
            ax.set_ylabel('Latitude grid')
        
        plt.tight_layout()
        plt.suptitle(f'Sea Ice Dataset Comparison - {date.strftime("%Y-%m-%d")}', 
                     fontsize=16, y=1.02)
        
        if save_plots:
            plt.savefig(f'sea_ice_comparison_{time_step}.png', dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图形释放内存

def main():
    """Main function to run all visualizations"""
    data_file = 'assets/sea_ice_dataset_withoutint.mat'
    
    print("Creating sea ice data visualizations...")
    
    # Overview of all datasets
    print("1. Creating overview plot...")
    visualize_sea_ice_overview(data_file)
    
    # Time series analysis
    print("2. Creating time series plot...")
    visualize_time_series(data_file)
    
    # Dataset comparison
    print("3. Creating comparison plot...")
    visualize_dataset_comparison(data_file, time_step=0)
    
    print("Visualizations complete!")

if __name__ == "__main__":
    main()