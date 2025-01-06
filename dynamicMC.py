import numpy as np
import matplotlib.pyplot as plt
import csv

import numpy as np

def simulate_laser_pulse_optimized(n_photons, lambda_rate, wx, wy, t_reset, pitch, xlim, ylim, pde, t_dist, spatial_dist):
    """
    Optimized simulation of a laser pulse, identifying pileup conditions.
    """
    # Generate photon arrival times and apply PDE
    t_arrivals = t_dist(size=n_photons, lambda_rate=lambda_rate)
    detected_mask = np.random.uniform(size=n_photons) < pde
    t_triggers = t_arrivals[detected_mask]

    # Generate spatial coordinates for detected photons
    x_positions, y_positions = spatial_dist(size=len(t_triggers), std_dev_x=wx / 4, std_dev_y=wy / 4)

    # Combine data into a single array
    photons = np.column_stack((x_positions, y_positions, t_triggers))

    # Sort photons by time
    photons = photons[np.argsort(photons[:, 2])]

    # Assign photons to spatial grid cells
    x_grid = np.floor((photons[:, 0] + xlim / 2) / pitch).astype(int)
    y_grid = np.floor((photons[:, 1] + ylim / 2) / pitch).astype(int)
    grid_indices = x_grid * (int(xlim / pitch) + 1) + y_grid

    # Initialize pileup and no_pileup masks
    is_pileup = np.zeros(len(photons), dtype=bool)

    # Vectorized pileup detection
    for i in range(len(photons)):
        # Print a status update every 10% of the way
        if i % (len(photons) // 10) == 0:
            print(f"Progress: {i / len(photons) * 100:.0f}%")
        if is_pileup[i]:
            continue

        # Select photons within the reset time window
        time_differences = photons[:, 2] - photons[i, 2]
        temporal_mask = (time_differences > 0) & (time_differences < t_reset)

        # Select photons in the same spatial cell
        same_cell_mask = grid_indices == grid_indices[i]

        # Combine masks
        pileup_candidates = temporal_mask & same_cell_mask
        is_pileup[pileup_candidates] = True

    # Separate pileup and no_pileup photons
    pileup = photons[is_pileup]
    no_pileup = photons[~is_pileup]

    return photons, pileup, no_pileup


def simulate_laser_pulse(n_photons, lambda_rate, wx, wy, t_reset, pitch, xlim, ylim, pde, t_dist, spatial_dist):
    """
    Simulates a laser pulse as a 3D object (x, y, t) and determines pileup conditions.

    Parameters:
        n_photons (int): Number of photons to simulate.
        t_dist (callable): Function to generate photon arrival times (e.g., np.random.uniform or np.random.normal).
        spatial_dist (callable): Function to generate (x, y) positions of photons (e.g., 2D Gaussian or uniform distribution).
        t_reset (float): Time reset threshold for pileup condition.

    Returns:
        photons (list): List of tuples [(x, y, t), ...] representing all photon arrivals.
        pileup (list): List of tuples [(x, y, t), ...] where pileup condition is true.
        no_pileup (list): List of tuples [(x, y, t), ...] where pileup condition is false.
    """
    # Generate photon arrival times using the temporal distribution
    t_arrivals = t_dist(size=n_photons, lambda_rate=lambda_rate)
    def quickplot():
        plt.hist(np.diff(t_arrivals), bins=np.logspace(-20, 0, 100))
        plt.xlabel('Time (s)')
        plt.ylabel('Number of photons')
        plt.title('Photon Arrival Times')
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
    #quickplot()

    # Generate spatial coordinates for each photon
    x_positions, y_positions = spatial_dist(size=n_photons, std_dev_x=wx/4, std_dev_y=wy/4)

    # Plot the spatial exposure
    def plot_spatial_exposure(x_positions, y_positions, xlim, ylim, pitch):
        """
        Plot the spatial exposure of photons.

        Parameters:
        x_positions (array-like): X coordinates of the photons.
        y_positions (array-like): Y coordinates of the photons.

        Returns:
        None
        """
        # Make bins using the pitch
        x_bins = np.arange(-xlim/2, xlim/2, pitch)
        y_bins = np.arange(-ylim/2, ylim/2, pitch)
        plt.figure(figsize=(8, 6))
        plt.hist2d(x_positions, y_positions, bins=[x_bins, y_bins], cmap='viridis')
        plt.colorbar(label='Number of photons')
        plt.xlabel('X position')
        plt.ylabel('Y position')
        plt.title('Spatial Exposure of Photons')
        plt.tight_layout()
        plt.show()
    #plot_spatial_exposure(x_positions, y_positions, xlim, ylim, pitch)

    # Combine into a single list of (x, y, t)
    photons = list(zip(x_positions, y_positions, t_arrivals))

    pileup = []
    no_pileup = []

    # Create lattice for spatial cells
    x_bins = np.arange(-xlim/2, xlim/2 + pitch, pitch)
    y_bins = np.arange(-ylim/2, ylim/2 + pitch, pitch)

    def find_cell(x, y):
        """Find the lattice cell indices for a given (x, y) coordinate."""
        x_idx = np.digitize(x, x_bins) - 1
        y_idx = np.digitize(y, y_bins) - 1
        return x_idx, y_idx

    # Check for pileup conditions
    looping = True
    i = 1
    skip = []
    while looping == True:

        # Skip if already marked as pileup
        if i in skip:
            i += 1
            if i == len(photons) - 1:
                looping = False
            else:
                continue

        # Check the PDE
        if np.random.uniform() > pde:
            i += 1
            if i == len(photons) - 1:
                looping = False
            continue

        x_i, y_i, t_i = photons[i]
        is_pileup = False
        cell_i = find_cell(x_i, y_i)

        for j in range(i + 1, len(photons)):

            x_j, y_j, t_j = photons[j]

            # If time difference exceeds reset time, break early
            if t_j - t_i >= t_reset:
                is_pileup = False
                break

            # Check if within twice the pitch
            if abs(x_i - x_j) < pitch and abs(y_i - y_j) < pitch:
                # Check if within the same spatial cell
                cell_j = find_cell(x_j, y_j)
                if cell_i == cell_j:
                    if np.random.uniform() < pde:
                        ispileup = True
                        pileup.append(photons[j])
                    skip.append(j)

        if is_pileup:
            pileup.append(photons[i])
        else:
            no_pileup.append(photons[i])
            i += 1
            if i == len(photons) - 1:
                looping = False
        

    return photons, pileup, no_pileup

# Temporal intensity distribution (Poisson distributed inter-arrival times)
def t_dist(size=1000, lambda_rate=10):
    inter_arrival_times = np.random.exponential(1 / lambda_rate, size=size)
    return np.cumsum(inter_arrival_times)  # Convert to cumulative arrival times

# Spatial distribution (2D Gaussian)
def spatial_dist(size=1000, std_dev_x=1E-3, std_dev_y=1E-3, mean=0):
    x_positions = np.random.normal(mean, std_dev_x, size=size)
    y_positions = np.random.normal(mean, std_dev_y, size=size)
    return x_positions, y_positions
    
# Example usage
if __name__ == "__main__":
    # Define parameters
    def setparam():
        # Define parameters
        n_photons = int(1E5)  # Number of photons to simulate
        t_reset = 5E-8  # s; reset time of the SPAD
        pitch = 50E-6  # m; Pitch between cells
        xlim = 5.95E-3  # m; X-axis limit for spatial distribution
        ylim = 5.85E-3  # m; Y-axis limit for spatial distribution
        wx = 1E-3  # m; Beam waist in x-direction
        wy = 1E-3  # m; Beam waist in y-direction
        pde = 0.3 # PDE of the SiPM
        lowrate = 6 # Logarithm of the lowest incident rate
        highrate = 13 # Logarithm of the highest incident rate
        n_processes = 10  # Number of parallel processes
        measured = [] # Measured rates
        return n_photons, t_reset, pitch, xlim, ylim, wx, wy, pde, lowrate, highrate, n_processes, measured
    n_photons, t_reset, pitch, xlim, ylim, wx, wy, pde, lowrate, highrate, n_processes, measured = setparam()
    incident = np.logspace(lowrate, highrate, num=int(highrate - lowrate + 1))
    params = {
        "n_photons": n_photons,
        "t_reset": t_reset,
        "pitch": pitch,
        "xlim": xlim,
        "ylim": ylim,
        "wx": wx,
        "wy": wy,
        "pde": pde,
        "lowrate": lowrate,
        "highrate": highrate,
        "n_processes": n_processes,
        "measured": measured
    }

    # Write simulation results to a CSV file
    def writeout(filename=None, data=None):
        """
        Write simulation results to a CSV file. If a filename is provided, write to that file.
        Otherwise, write to 'simulation_results.csv'.

        Parameters:
        filename (str, optional): The name of the file to write to. Defaults to None.

        Returns:
        None
        """
        if filename is None:
            filename = "simulation_results.csv"
            with open(filename, mode='w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)

                # Write header
                csvwriter.writerow([f"{key}: {value}" for key, value in params.items()])
                csvwriter.writerow(["Incident Rate (cps)", "Measured Rate (cps)"])
            return filename
        
        if filename is not None and data is not None:
            with open(filename, mode='a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                # Save results to CSV
                dout = [data[i] for i in range(len(data))]
                csvwriter.writerow(dout)
            return filename, dout   
        
        return None
    output_filename = writeout()

    for lambda_rate in incident:
        # Print the progress
        duration = n_photons/lambda_rate  # s; Duration of the simulation
        while duration < 10*t_reset:
            n_photons = n_photons * 5
            duration = n_photons/lambda_rate
            print(f"Duration ({duration:.2e} s) too short for pileup. Increasing the number of photons to {n_photons}")
        
        print(f"Exposure duration: {duration:.2e} s")
        print(f"Photon rate: {lambda_rate:.2e} cps")

        # Simulate the laser pulse 
        photons, pileup, no_pileup = simulate_laser_pulse_optimized(n_photons, lambda_rate, wx, wy, t_reset, pitch, xlim, ylim, pde, t_dist, spatial_dist)
        # Output results
        #print("\n")
        #print(f"Total photons: {len(photons)}")
        #print(f"Pileup avalanches: {len(pileup)}")
        #print(f"Measured avalanches: {len(no_pileup)}")
        #print(f"Measured rate: {len(no_pileup) / duration:.2e} cps")

        measured.append(len(no_pileup) / duration)
        output_filename, rates = writeout(output_filename, [lambda_rate, measured[-1]])
    
    # Plot the results
    figure, axis = plt.subplots()
    axis.plot(incident, measured, 'o-')
    axis.set_xscale('log')
    axis.set_yscale('log')
    axis.set_xlabel('Incident Rate (cps)')
    axis.set_ylabel('Measured Rate (cps)')
    plt.show()

    print(incident)
    print(measured)



