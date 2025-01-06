import numpy as np
import matplotlib.pyplot as plt
import csv
import multiprocessing
import time

def simulate_laser_pulse(n_photons, lambda_rate, wx, wy, t_reset, pitch, xlim, ylim, pde, t_dist, spatial_dist):
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

def process_batch(args):
    """Helper function to process a batch of photons."""
    return simulate_laser_pulse(*args)

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
    t0 = time.time()
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
        highrate = 14 # Logarithm of the highest incident rate
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
            n_photons = n_photons * 10
            print(f"Duration ({duration:.2e} s) too short for pileup. Increasing the number of photons to {n_photons}")
            duration = n_photons/lambda_rate

        print(f"Exposure duration: {duration:.2e} s")
        print(f"Photon rate: {lambda_rate:.2e} cps")

        # Divide photons into batches for multiprocessing
        batch_size = n_photons // n_processes
        args = [(batch_size, lambda_rate, wx, wy, t_reset, pitch, xlim, ylim, pde, t_dist, spatial_dist) for _ in range(n_processes)]

        # Use multiprocessing to simulate laser pulse in parallel
        with multiprocessing.Pool(n_processes) as pool:
            results = pool.map(process_batch, args)

        # Combine results from all processes
        all_photons = []
        all_pileup = []
        all_no_pileup = []

        for photons, pileup, no_pileup in results:
            all_photons.extend(photons)
            all_pileup.extend(pileup)
            all_no_pileup.extend(no_pileup)

        measured.append(len(all_no_pileup) / duration)
        output_filename, rates = writeout(output_filename, [lambda_rate, measured[-1]])
    
    print(f"\nSimulation time: {time.time() - t0:.2f} s")

    # Plot the results
    figure, axis = plt.subplots()
    axis.plot(incident, measured, 'o-')
    axis.set_xscale('log')
    axis.set_yscale('log')
    axis.set_xlabel('Incident Rate (cps)')
    axis.set_ylabel('Measured Rate (cps)')
    plt.show()
    



