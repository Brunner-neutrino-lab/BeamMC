import numpy as np
import multiprocessing
import matplotlib.pyplot as plt

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

    # Generate spatial coordinates for each photon
    x_positions, y_positions = spatial_dist(size=n_photons, std_dev_x=wx / 4, std_dev_y=wy / 4)

    # Combine into a single list of (x, y, t)
    photons = list(zip(x_positions, y_positions, t_arrivals))

    pileup = []
    no_pileup = []

    # Create lattice for spatial cells
    x_bins = np.arange(-xlim / 2, xlim / 2 + pitch, pitch)
    y_bins = np.arange(-ylim / 2, ylim / 2 + pitch, pitch)

    def find_cell(x, y):
        """Find the lattice cell indices for a given (x, y) coordinate."""
        x_idx = np.digitize(x, x_bins) - 1
        y_idx = np.digitize(y, y_bins) - 1
        return x_idx, y_idx

    # Check for pileup conditions
    looping = True
    i = 1
    skip = []
    while looping:

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
                        is_pileup = True
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
    # Define parameters
    n_photons = int(1E5)  # Number of photons to simulate
    t_reset = 5E-8  # s; reset time of the SPAD
    pitch = 50E-6  # m; Pitch between cells
    xlim = 5.95E-3  # m; X-axis limit for spatial distribution
    ylim = 5.85E-3  # m; Y-axis limit for spatial distribution
    wx = 1E-3  # m; Beam waist in x-direction
    wy = 1E-3  # m; Beam waist in y-direction
    pde = 0.3
    lowrate = 6
    highrate = 13
    n_processes = 10  # Number of parallel processes

    # Iterate through different incident rates
    incident = np.logspace(lowrate, highrate, num=int(highrate - lowrate + 1))
    measured = []

    for lambda_rate in incident:
        # Print the progress
        duration = n_photons / lambda_rate  # s; Duration of the simulation
        if duration < 10 * t_reset:
            n_photons = int(10 * t_reset * lambda_rate)
            duration = n_photons / lambda_rate
            print(f"Duration ({duration:.2e} s) too short for pileup. Increasing the number of photons to {n_photons}")

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

        # Output results
        print("\n")
        print(f"Total photons: {len(all_photons)}")
        print(f"Pileup avalanches: {len(all_pileup)}")
        print(f"Measured avalanches: {len(all_no_pileup)}")
        print(f"Measured rate: {len(all_no_pileup) / duration:.2e} cps")

        measured.append(len(all_no_pileup) / duration)

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