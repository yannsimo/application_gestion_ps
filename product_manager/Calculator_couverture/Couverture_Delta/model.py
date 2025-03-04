import numpy as np
from numba import jit, prange
from numba.typed import List
from typing import Dict, Tuple
from datetime import datetime

def get_singleton_market_data():
    from ...Data.SingletonMarketData import SingletonMarketData
    return SingletonMarketData


@jit(nopython=True, fastmath=True)
def simulate_px_x_path(px0, x0, r_d, r_f, sigma_px, sigma_x, T, dt, Z_px, Z_x):
    """
    Simulate a single path for PX and X
    
    Args:
        px0: Initial PX value
        x0: Initial X value
        r_d: Domestic interest rate
        r_f: Foreign interest rate
        sigma_px: PX volatility
        sigma_x: X volatility
        T: Time horizon
        dt: Time step
        Z_px: Random shocks for PX
        Z_x: Random shocks for X
        
    Returns:
        Tuple containing PX and X paths
    """
    steps = int(T / dt) + 1
    px_path = np.zeros(steps, dtype=np.float64)
    x_path = np.zeros(steps, dtype=np.float64)
    
    # Set initial values
    px_path[0] = px0
    x_path[0] = x0
    
    # Simulation constants
    sqrt_dt = np.sqrt(dt)
    px_drift = (r_d - 0.5 * sigma_px**2) * dt
    x_drift = (r_d - r_f - 0.5 * sigma_x**2) * dt
    
    # Simulate path
    for i in range(1, steps):
        # Update PX
        px_diffusion = sigma_px * sqrt_dt * Z_px[i-1]
        px_path[i] = px_path[i-1] * np.exp(px_drift + px_diffusion)
        
        # Update X
        x_diffusion = sigma_x * sqrt_dt * Z_x[i-1]
        x_path[i] = x_path[i-1] * np.exp(x_drift + x_diffusion)
    
    return px_path, x_path


@jit(nopython=True, parallel=True, fastmath=True)
def simulate_px_x_paths(px0_array, x0_array, r_d_array, r_f_array, sigma_px_array, 
                        sigma_x_array, T, dt, num_paths, correlated_Z):
    """
    Simulate multiple paths for multiple PX and X processes
    
    Args:
        px0_array: Array of initial PX values
        x0_array: Array of initial X values
        r_d_array: Array of domestic interest rates
        r_f_array: Array of foreign interest rates
        sigma_px_array: Array of PX volatilities
        sigma_x_array: Array of X volatilities
        T: Time horizon
        dt: Time step
        num_paths: Number of simulation paths
        correlated_Z: Correlated random variables
        
    Returns:
        Tuple containing PX and X paths for all simulations
    """
    num_px = len(px0_array)
    num_x = len(x0_array)
    steps = int(T / dt) + 1
    
    # Initialize result arrays
    px_paths = np.zeros((num_px, num_paths, steps), dtype=np.float64)
    x_paths = np.zeros((num_x, num_paths, steps), dtype=np.float64)
    
    # Simulate paths
    for p in prange(num_paths):
        for i in range(num_px):
            Z_px = correlated_Z[i, p, :]
            px_paths[i, p, 0] = px0_array[i]
            
            for j in range(1, steps):
                # Update PX (simplified version without calling function to improve performance)
                px_drift = (r_d_array[i] - 0.5 * sigma_px_array[i]**2) * dt
                px_diffusion = sigma_px_array[i] * np.sqrt(dt) * Z_px[j-1]
                px_paths[i, p, j] = px_paths[i, p, j-1] * np.exp(px_drift + px_diffusion)
        
        for i in range(num_x):
            Z_x = correlated_Z[num_px + i, p, :]
            x_paths[i, p, 0] = x0_array[i]
            
            for j in range(1, steps):
                # Update X (simplified version without calling function to improve performance)
                x_drift = (r_d_array[i] - r_f_array[i] - 0.5 * sigma_x_array[i]**2) * dt
                x_diffusion = sigma_x_array[i] * np.sqrt(dt) * Z_x[j-1]
                x_paths[i, p, j] = x_paths[i, p, j-1] * np.exp(x_drift + x_diffusion)
    
    return px_paths, x_paths


@jit(nopython=True, parallel=True, fastmath=True)
def calculate_p_paths(px_paths, x_paths):
    """
    Calculate P paths from PX and X paths
    
    Args:
        px_paths: Simulated PX paths
        x_paths: Simulated X paths
        
    Returns:
        P paths
    """
    num_indices = px_paths.shape[0]
    num_paths = px_paths.shape[1]
    num_steps = px_paths.shape[2]
    
    # Initialize P paths
    p_paths = np.zeros_like(px_paths)
    
    # Calculate P = PX / X
    for i in prange(num_indices):
        x_idx = min(i, x_paths.shape[0] - 1)  # Handle indices that use EUR (no exchange rate)
        
        for p in range(num_paths):
            for t in range(num_steps):
                if x_paths[x_idx, p, t] > 0:  # Avoid division by zero
                    p_paths[i, p, t] = px_paths[i, p, t] / x_paths[x_idx, p, t]
    
    return p_paths


class BlackScholesSimulation:
    """
    Class for simulating price paths using Black-Scholes model
    
    This updated version simulates both PX and X processes simultaneously
    with proper correlation structure
    """
    @staticmethod
    def simulate_px_x(px0_array, x0_array, r_d_array, r_f_array, sigma_px_array, 
                     sigma_x_array, T, dt, num_paths, cholesky_matrix):
        """
        Simulate PX and X paths
        
        Args:
            px0_array: Array of initial PX values
            x0_array: Array of initial X values
            r_d_array: Array of domestic interest rates
            r_f_array: Array of foreign interest rates
            sigma_px_array: Array of PX volatilities
            sigma_x_array: Array of X volatilities
            T: Time horizon
            dt: Time step
            num_paths: Number of simulation paths
            cholesky_matrix: Cholesky decomposition of correlation matrix
            
        Returns:
            Tuple containing PX, X, and P paths
        """
        # Number of time steps
        steps = int(T / dt) + 1
        
        # Number of processes
        num_processes = len(px0_array) + len(x0_array)
        
        # Generate independent random variables
        Z = np.random.standard_normal((num_processes, num_paths, steps - 1))
        
        # Apply correlation structure
        correlated_Z = np.zeros_like(Z)
        for t in range(steps - 1):
            for p in range(num_paths):
                # Reshape Z[t, p] to a column vector
                z_vec = Z[:, p, t].reshape(-1, 1)
                
                # Apply Cholesky decomposition
                correlated_vec = np.dot(cholesky_matrix, z_vec).flatten()
                
                # Store correlated random variables
                correlated_Z[:, p, t] = correlated_vec
        
        # Simulate paths
        px_paths, x_paths = simulate_px_x_paths(
            px0_array, x0_array, r_d_array, r_f_array,
            sigma_px_array, sigma_x_array, T, dt, num_paths, correlated_Z
        )
        
        # Calculate P paths
        p_paths = calculate_p_paths(px_paths, x_paths)
        
        return px_paths, x_paths, p_paths