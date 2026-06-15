"""
SPH Visualization Module

A flexible, high-performance module for SPH (Smoothed Particle Hydrodynamics) visualization.
This module creates 2D projections from particle data for scientific visualization using 
proper SPH kernel methods.

Main features:
- High-performance CPU rendering using Numba with parallel processing
- Proper SPH kernel implementation (cubic spline, quintic spline, Wendland C4)
- Memory-efficient processing for large datasets via chunking and tiling
- Support for density and field visualizations
- Mass conservation checking and correction
- Precomputed kernel grids for performance optimization
- Support for periodic boundary conditions
"""

import os
import time
import warnings
import multiprocessing
from typing import Tuple, Optional, Callable, Union, List, Dict, Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from numba import njit, prange, set_num_threads, get_num_threads
import concurrent.futures

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
try:
    import imageio
except ImportError:
    imageio = None

# Advanced imports for spatial acceleration
try:
    import numba.typed as typed
    NUMBA_TYPED_AVAILABLE = True
except ImportError:
    NUMBA_TYPED_AVAILABLE = False


# Get number of threads from environment or use a reasonable default
def _get_default_nthreads() -> int:
    """Get the default number of threads to use for parallel processing."""
    env_threads = os.environ.get('NUMBA_NUM_THREADS') or os.environ.get('OMP_NUM_THREADS')
    if env_threads:
        try:
            return int(env_threads)
        except ValueError:
            pass
    # Default to number of CPU cores, leaving one core free
    return max(1, multiprocessing.cpu_count() - 1)


# SPH Kernels
@njit(nopython=True, fastmath=True, cache=True, nogil=True)
def cubic_spline_kernel(r: float, h: float) -> float:
    """
    Cubic spline SPH kernel.
    
    Parameters
    ----------
    r : float
        Distance
    h : float
        Smoothing length
        
    Returns
    -------
    float
        Kernel value
    """
    q = r / h
    
    if q > 1.0:
        return 0.0
    
    factor = 8.0 / (np.pi * h**3)
    
    if q <= 0.5:
        return factor * (1.0 - 6.0 * q**2 + 6.0 * q**3)
    else:  # 0.5 < q <= 1.0
        return factor * 2.0 * (1.0 - q)**3
    
@njit(nopython=True, fastmath=True, cache=True, nogil=True)
def cubic_spline_kernel_2D_proj(r: float, h: float) -> float:
    """
    Cubic spline SPH kernel for 2D projection with analytical integration.
    
    This kernel provides an optimized 2D projection of the cubic spline kernel
    with analytical integration, suitable for rendering SPH data onto 2D images.
    
    Parameters
    ----------
    r : float
        Distance from particle center
    h : float
        Smoothing length
        
    Returns
    -------
    float
        Kernel value for 2D projection
    """
    if h <= 0.0:
        return 0.0
    
    q = r / h
    q2 = q * q
    
    if q2 >= 1.0:
        return 0.0
    
    # Normalization factor
    fac = 1.0 / (1.00023 * h * h)
    
    if q2 < 0.25:
        # For q < 0.5: optimized polynomial evaluation
        q4 = q2 * q2
        return fac * (1.909859317102744 - 10.23669021 * q2 - 23.27182034 * q4 * (q2 - 1.0))
    else:
        # For 0.5 <= q < 1.0: full polynomial evaluation
        q4 = q2 * q2
        q3 = q2 * np.sqrt(q2)  # q^3 = q^2 * sqrt(q^2)
        
        return fac * (-0.2475743559207261 * q4 
                     - 3.556986664656963 * q3 
                     + 11.67894130021956 * q2 
                     - 11.71909411592771 * np.sqrt(q2) 
                     + 3.84471383628584)


@njit(nopython=True, fastmath=True, cache=True, nogil=True)
def quintic_spline_kernel(r: float, h: float) -> float:
    """
    Quintic spline SPH kernel with compact support.
    
    Parameters
    ----------
    r : float
        Distance
    h : float
        Smoothing length
        
    Returns
    -------
    float
        Kernel value
    """
    q = r / h
    
    if q > 1.0:
        return 0.0
    
    factor = 1.0 / (120.0 * np.pi * h**3)
    
    return factor * (1.0 - q)**5 * (8.0 + 40.0*q + 48.0*q**2 + 16.0*q**3)


@njit(nopython=True, fastmath=True, cache=True, nogil=True)
def wendland_c4_kernel(r: float, h: float) -> float:
    """
    Wendland C4 kernel - higher order, suitable for convergence studies.
    
    Parameters
    ----------
    r : float
        Distance
    h : float
        Smoothing length
        
    Returns
    -------
    float
        Kernel value
    """
    q = r / h
    
    if q > 1.0:
        return 0.0
    
    factor = 495.0 / (32.0 * np.pi * h**3)
    
    return factor * (1.0 - q)**6 * (1.0 + 6.0*q + 35.0/3.0*q**2)


# Map kernel names to functions
KERNELS = {
    'cubic': cubic_spline_kernel_2D_proj,
    'quintic': quintic_spline_kernel,
    'wendland': wendland_c4_kernel,
}


# Vectorized kernel operations for SIMD acceleration
@njit(nopython=True, fastmath=True, cache=True, nogil=True)
def vectorized_kernel_evaluation(
    distances: np.ndarray,
    hs: np.ndarray,
    kernel_func_id: int,
    pixel_area: float
) -> np.ndarray:
    """
    Vectorized kernel evaluation for multiple particles at once.
    
    Parameters
    ----------
    distances : np.ndarray
        Distance array, shape (N,)
    hs : np.ndarray
        Smoothing lengths, shape (N,)
    kernel_func_id : int
        Kernel function ID
    pixel_area : float
        Pixel area for normalization
        
    Returns
    -------
    np.ndarray
        Kernel values
    """
    n = distances.shape[0]
    kernel_values = np.zeros(n, dtype=np.float32)
    
    # Vectorized computation
    for i in range(n):
        if distances[i] < hs[i] * 2.0:  # kernel support
            if kernel_func_id == 0:
                kernel_values[i] = cubic_spline_kernel_2D_proj(distances[i], hs[i]) * pixel_area
            elif kernel_func_id == 1:
                kernel_values[i] = quintic_spline_kernel(distances[i], hs[i]) * pixel_area
            elif kernel_func_id == 2:
                kernel_values[i] = wendland_c4_kernel(distances[i], hs[i]) * pixel_area
            else:
                kernel_values[i] = cubic_spline_kernel_2D_proj(distances[i], hs[i]) * pixel_area
                
    return kernel_values


@njit(nopython=True, fastmath=True, cache=True, nogil=True, parallel=True)
def fast_distance_computation(
    pos1: np.ndarray,
    pos2: np.ndarray
) -> np.ndarray:
    """
    Fast distance computation using optimized operations.
    
    Parameters
    ----------
    pos1 : np.ndarray
        First set of positions, shape (N, 2)
    pos2 : np.ndarray
        Second set of positions, shape (M, 2)
        
    Returns
    -------
    np.ndarray
        Distance matrix, shape (N, M)
    """
    n1, n2 = pos1.shape[0], pos2.shape[0]
    distances = np.zeros((n1, n2), dtype=np.float32)
    
    for i in range(n1):
        for j in range(n2):
            dx = pos1[i, 0] - pos2[j, 0]
            dy = pos1[i, 1] - pos2[j, 1]
            distances[i, j] = np.sqrt(dx*dx + dy*dy)
    
    return distances


# Memory pool for reducing allocations
class MemoryPool:
    """Simple memory pool to reduce allocations during rendering."""
    
    def __init__(self):
        self.float32_arrays = []
        self.int32_arrays = []
    
    def get_float32_array(self, size: int) -> np.ndarray:
        """Get a float32 array from the pool."""
        for i, arr in enumerate(self.float32_arrays):
            if arr.size >= size:
                result = arr[:size]
                result.fill(0)
                self.float32_arrays.pop(i)
                return result
        
        return np.zeros(size, dtype=np.float32)
    
    def get_int32_array(self, size: int) -> np.ndarray:
        """Get an int32 array from the pool."""
        for i, arr in enumerate(self.int32_arrays):
            if arr.size >= size:
                result = arr[:size]
                result.fill(0)
                self.int32_arrays.pop(i)
                return result
        
        return np.zeros(size, dtype=np.int32)
    
    def return_array(self, arr: np.ndarray):
        """Return an array to the pool."""
        if arr.dtype == np.float32:
            self.float32_arrays.append(arr)
        elif arr.dtype == np.int32:
            self.int32_arrays.append(arr)


# Global memory pool instance
_memory_pool = MemoryPool()


@njit(nopython=True, fastmath=True, cache=True, nogil=True)
def compute_pixel_boundaries(
    pos_x: float, 
    pos_y: float, 
    h: float, 
    kernel_radius: float,
    xmin: float, 
    ymin: float, 
    pixel_size_x: float, 
    pixel_size_y: float, 
    nx: int, 
    ny: int
) -> Tuple[int, int, int, int]:
    """
    Compute the boundary of pixels affected by a particle.
    
    Parameters
    ----------
    pos_x, pos_y : float
        Particle position
    h : float
        Smoothing length
    kernel_radius : float
        Multiple of h defining support radius
    xmin, ymin : float
        Lower bounds of the image
    pixel_size_x, pixel_size_y : float
        Size of a pixel in data units
    nx, ny : int
        Image dimensions
        
    Returns
    -------
    Tuple[int, int, int, int]
        Pixel boundaries (imin, imax, jmin, jmax)
    """
    support = kernel_radius * h
    
    # Calculate pixel coordinates for boundaries
    imin = max(0, int((pos_x - support - xmin) / pixel_size_x))
    imax = min(nx-1, int((pos_x + support - xmin) / pixel_size_x) + 1)
    jmin = max(0, int((pos_y - support - ymin) / pixel_size_y))
    jmax = min(ny-1, int((pos_y + support - ymin) / pixel_size_y) + 1)
    
    return imin, imax, jmin, jmax


@njit(nopython=True, fastmath=True, cache=True, nogil=True)
def bilinear_interpolate(grid: np.ndarray, x: float, y: float) -> float:
    """
    Perform bilinear interpolation on a 2D grid.
    
    Parameters
    ----------
    grid : np.ndarray
        2D array to interpolate from
    x, y : float
        Coordinates to interpolate at (can be fractional)
        
    Returns
    -------
    float
        Interpolated value
    """
    h, w = grid.shape
    
    # Clamp coordinates to valid range
    x = max(0.0, min(x, w - 1.001))
    y = max(0.0, min(y, h - 1.001))
    
    # Get integer and fractional parts
    x0 = int(x)
    y0 = int(y)
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    
    # Fractional parts
    fx = x - x0
    fy = y - y0
    
    # Bilinear interpolation
    val = (grid[y0, x0] * (1.0 - fx) * (1.0 - fy) +
           grid[y0, x1] * fx * (1.0 - fy) +
           grid[y1, x0] * (1.0 - fx) * fy +
           grid[y1, x1] * fx * fy)
    
    return val


@njit(nopython=True, fastmath=True, cache=True, nogil=True)
def bilinear_interpolate_region(
    grid: np.ndarray, 
    x_start: float, 
    y_start: float, 
    x_end: float, 
    y_end: float,
    n_samples: int = 4
) -> float:
    """
    Compute the integrated kernel contribution over a region using bilinear interpolation.
    
    This function samples the grid at multiple points within the region and averages
    the interpolated values, scaled by the region area relative to grid cell area.
    
    Parameters
    ----------
    grid : np.ndarray
        2D SPH kernel grid
    x_start, y_start : float
        Start coordinates of the region (fractional grid indices)
    x_end, y_end : float
        End coordinates of the region (fractional grid indices)
    n_samples : int
        Number of sample points per dimension for integration
        
    Returns
    -------
    float
        Integrated kernel contribution for the region
    """
    h, w = grid.shape
    
    # Clamp to valid range
    x_start = max(0.0, min(x_start, w - 0.001))
    x_end = max(0.0, min(x_end, w - 0.001))
    y_start = max(0.0, min(y_start, h - 0.001))
    y_end = max(0.0, min(y_end, h - 0.001))
    
    # Region dimensions in grid units
    region_width = x_end - x_start
    region_height = y_end - y_start
    
    if region_width <= 0.0 or region_height <= 0.0:
        return 0.0
    
    # For very small regions, just use center point interpolation
    if region_width < 1.0 and region_height < 1.0:
        cx = (x_start + x_end) * 0.5
        cy = (y_start + y_end) * 0.5
        return bilinear_interpolate(grid, cx, cy) * region_width * region_height
    
    # For larger regions, use adaptive sampling
    # Determine number of samples based on region size
    nx_samples = max(2, min(n_samples, int(region_width) + 1))
    ny_samples = max(2, min(n_samples, int(region_height) + 1))
    
    # Sample spacing
    dx = region_width / nx_samples
    dy = region_height / ny_samples
    
    # Integrate using sampled points
    total = 0.0
    for i in range(ny_samples):
        y = y_start + (i + 0.5) * dy
        for j in range(nx_samples):
            x = x_start + (j + 0.5) * dx
            total += bilinear_interpolate(grid, x, y) * dx * dy    

    return total


@njit(nopython=True, fastmath=True, cache=True, nogil=True)
def calculate_kernel_contribution(
    pos_x: float,
    pos_y: float,
    h_i: float,
    pixel_bounds: Tuple[int, int, int, int],
    xmin: float,
    ymin: float,
    pixel_size_x: float,
    pixel_size_y: float,
    pixel_size: float,
    pixel_size_full: float,
    kernel_func_id: int,
    kernel_radius: float,
    sph_grid: Optional[np.ndarray] = None,
    use_bilinear: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Calculate the kernel contribution for a particle on a grid of pixels.
    
    Parameters
    ----------
    pos_x, pos_y : float
        Particle position
    h_i : float
        Smoothing length
    pixel_bounds : Tuple[int, int, int, int]
        Pixel boundaries (imin, imax, jmin, jmax)
    xmin, ymin : float
        Lower bounds of the image
    pixel_size_x, pixel_size_y, pixel_size : float
        Size of a pixel in data units
    kernel_func_id : int
        ID of kernel function (0: cubic, 1: quintic, 2: wendland)
    kernel_radius : float
        Multiple of h defining support radius
    sph_grid : np.ndarray, optional
        Precomputed SPH kernel grid
    use_bilinear : bool, default=True
        If True, use bilinear interpolation for grid lookups when boundaries
        don't align with grid cells. Provides smoother results at slight
        computational cost.
        
    Returns
    -------
    Tuple[np.ndarray, float]
        Array of kernel values and their total sum for normalization
    """
    imin, imax, jmin, jmax = pixel_bounds
    pixel_area = pixel_size_x * pixel_size_y
    
    # Create array for kernel values
    n_pixels = (imax - imin + 1) * (jmax - jmin + 1)
    kernel_values = np.zeros(n_pixels, dtype=np.float32)
    
    # Check if grid approximation should be used
    grid_resolution = 0
    if sph_grid is not None:
        grid_resolution = sph_grid.shape[0]
    
    # Use grid for medium-sized particles (following Gaepsi2 approach)
    use_grid = sph_grid is not None and h_i < pixel_size_full and grid_resolution > 0
    
    # Calculate total kernel weight for normalization (similar to Gaepsi2's 'bit')
    total_weight = 0.0
    
    if h_i < pixel_size_full:
        # For small to medium particles, use actual kernel calculation
        # h_half = h_i * 0.5
        cell_size = h_i / grid_resolution if use_grid else 0.0
        
        # Precompute particle position components
        pos_x_min = pos_x - h_i
        pos_y_min = pos_y - h_i
        pos_x_max = pos_x + h_i
        pos_y_max = pos_y + h_i

        # Compute kernel values for all affected pixels
        s = 0
        for j in range(jmin, jmax + 1):
            pixel_y = ymin + ((j + 0.5) * pixel_size_y)
            
            for k in range(imin, imax + 1):
                pixel_x = xmin + ((k + 0.5) * pixel_size_x)
                
                if use_grid:
                    # Fast grid-based approximation 
                    pixel_x_min = pixel_x - (pixel_size_x * 0.5)
                    pixel_y_min = pixel_y - (pixel_size_y * 0.5)
                    pixel_x_max = pixel_x + (pixel_size_x * 0.5)
                    pixel_y_max = pixel_y + (pixel_size_y * 0.5)

                    
                    # Compute overlap region
                    ox = max(pos_x_min, pixel_x_min)
                    oy = max(pos_y_min, pixel_y_min)
                    ow = min(pos_x_max, pixel_x_max) - ox
                    oh = min(pos_y_max, pixel_y_max) - oy
                    
                    if ow > 0.0 and oh > 0.0:
                        # Map to fractional grid indices for bilinear interpolation
                        x_start_f = (ox - pos_x_min) / cell_size
                        y_start_f = (oy - pos_y_min) / cell_size
                        x_end_f = (ox + ow - pos_x_min) / cell_size
                        y_end_f = (oy + oh - pos_y_min) / cell_size
                        
                        if use_bilinear:
                            # Use bilinear interpolation for smoother results
                            kval = bilinear_interpolate_region(
                                sph_grid, x_start_f, y_start_f, x_end_f, y_end_f
                            )
                        else:
                            # Original integer-based grid lookup
                            x_start = max(0, int(x_start_f))
                            y_start = max(0, int(y_start_f))
                            x_end = min(grid_resolution, int(x_end_f) + 1)
                            y_end = min(grid_resolution, int(y_end_f) + 1)
                            
                            # Sum kernel values from grid
                            if x_end > x_start and y_end > y_start:
                                kval = np.sum(sph_grid[y_start:y_end, x_start:x_end])
                            else:
                                kval = 0.0
                        
                        kernel_values[s] = kval
                        total_weight += kval
                else:
                    # Direct kernel calculation
                    dx = pixel_x - pos_x
                    dy = pixel_y - pos_y
                    r2 = dx*dx + dy*dy
                    
                    if r2 < (h_i * kernel_radius) ** 2:
                        r = np.sqrt(r2)
                        
                        # Apply kernel function based on ID
                        if kernel_func_id == 0:
                            kval = cubic_spline_kernel_2D_proj(r, h_i)
                        elif kernel_func_id == 1:
                            kval = quintic_spline_kernel(r, h_i)
                        elif kernel_func_id == 2:
                            kval = wendland_c4_kernel(r, h_i)
                        else:
                            kval = cubic_spline_kernel_2D_proj(r, h_i)
                        
                        kval *= pixel_area
                        kernel_values[s] = kval
                        total_weight += kval
                
                s += 1
    else:
        # For large particles, do direct kernel calculation without grid
        s = 0
        for j in range(jmin, jmax + 1):
            pixel_y = ymin + ((j + 0.5) * pixel_size_y)
            
            for k in range(imin, imax + 1):
                pixel_x = xmin + ((k + 0.5) * pixel_size_x)
                
                dx = pixel_x - pos_x
                dy = pixel_y - pos_y
                r2 = dx*dx + dy*dy
                
                if r2 < (h_i * kernel_radius) ** 2:
                    r = np.sqrt(r2)
                    
                    # Apply kernel function based on ID
                    if kernel_func_id == 0:
                        kval = cubic_spline_kernel_2D_proj(r, h_i)
                    elif kernel_func_id == 1:
                        kval = quintic_spline_kernel(r, h_i)
                    elif kernel_func_id == 2:
                        kval = wendland_c4_kernel(r, h_i)
                    else:
                        kval = cubic_spline_kernel_2D_proj(r, h_i)
                    
                    kval *= pixel_area
                    kernel_values[s] = kval
                    total_weight += kval
                
                s += 1

    return kernel_values, total_weight




@njit(nopython=True, fastmath=True, cache=True, nogil=True, parallel=True)
def render_projection_chunk_numba(
    positions: np.ndarray,
    masses: np.ndarray,
    hs: np.ndarray,
    field: Optional[np.ndarray],
    xmin: float,
    ymin: float,
    pixel_size_x: float,
    pixel_size_y: float,
    nx: int,
    ny: int,
    kernel_func_id: int,
    kernel_radius: float,
    sph_grid: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Render a chunk of particles using Numba parallelization.
    
    Parameters
    ----------
    positions : np.ndarray
        Particle positions, shape (N, 3)
    masses : np.ndarray
        Particle masses, shape (N,)
    hs : np.ndarray
        Smoothing lengths, shape (N,)
    field : np.ndarray or None
        Optional field values, shape (N,)
    xmin, ymin : float
        Lower bounds of the image
    pixel_size_x, pixel_size_y : float
        Size of a pixel in data units
    nx, ny : int
        Image dimensions
    kernel_func_id : int
        ID of kernel function (0: cubic, 1: quintic, 2: wendland)
    kernel_radius : float
        Multiple of h defining support radius
    sph_grid : np.ndarray, optional
        Precomputed SPH kernel grid
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Rendered image and weight map
    """
    n_particles = positions.shape[0]
    
    # Initialize image and weights arrays
    image = np.zeros((ny, nx), dtype=np.float32)
    weights = np.zeros((ny, nx), dtype=np.float32)
    
    # Precompute the maximum pixel size for performance
    pixel_size = max(pixel_size_x, pixel_size_y)
    pixel_size_half = pixel_size * 1e-10
    pixel_size_full = pixel_size * 1e10

    # Process particles in parallel
    for i in prange(n_particles):
        pos_x, pos_y = positions[i, 0], positions[i, 1]
        h_i = max(hs[i], 1e-10)  # Avoid zero smoothing length
        m_i = masses[i]
        field_val = 0.0 if field is None else field[i]
        
        # Very small smoothing length optimization
        if h_i < pixel_size_half:
            # Deposit to nearest pixel only
            k = int((pos_x - xmin) / pixel_size_x)
            j = int((pos_y - ymin) / pixel_size_y)
            
            # Ensure indices are within bounds
            if 0 <= k < nx and 0 <= j < ny:
                w = m_i
                weights[j, k] += w
                
                if field is not None:
                    image[j, k] += w * field_val
                else:
                    image[j, k] += w
            
            # Skip the rest of the loop for small particles
            continue
        
        # Compute pixel boundaries for this particle
        imin, imax, jmin, jmax = compute_pixel_boundaries(
            pos_x, pos_y, h_i, kernel_radius, 
            xmin, ymin, pixel_size_x, pixel_size_y, nx, ny
        )
        
        # Skip if particle doesn't affect any pixels
        if imin > imax or jmin > jmax:
            continue
        
        # Calculate kernel contribution using the separate function
        kernel_values, total_weight = calculate_kernel_contribution(
            pos_x, pos_y, h_i,
            (imin, imax, jmin, jmax),
            xmin, ymin, pixel_size_x, pixel_size_y, pixel_size, pixel_size_full,
            kernel_func_id, kernel_radius, sph_grid
        )
        
        # Normalize kernel weights (Gaepsi2-style)
        if total_weight > 0.0:
            norm_factor = 1.0 / total_weight
        else:
            norm_factor = 0.0
        
        # Apply normalized kernel weights to image
        s = 0
        for j in range(jmin, jmax + 1):
            for k in range(imin, imax + 1):
                w = kernel_values[s] # * norm_factor
                s += 1
                
                if w > 0.0:
                    w_m = m_i * w
                    weights[j, k] += w_m
                    
                    if field is not None:
                        image[j, k] += w_m * field_val
                    else:
                        image[j, k] += w_m
    
    return image, weights


def process_chunk_parallel(
    chunk_data: Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]],
    rendering_params: Dict[str, Any],
    sph_grid: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process a single chunk for parallel execution with concurrent futures.
    
    Parameters
    ----------
    chunk_data : Tuple
        (chunk_pos, chunk_masses, chunk_hs, chunk_field)
    rendering_params : Dict
        Dictionary containing rendering parameters
    sph_grid : np.ndarray, optional
        Precomputed SPH kernel grid
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Rendered chunk image and weight map
    """
    chunk_pos, chunk_masses, chunk_hs, chunk_field = chunk_data
    
    # Extract parameters and ensure correct types
    xmin = float(rendering_params['xmin'])
    ymin = float(rendering_params['ymin'])
    pixel_size_x = float(rendering_params['pixel_size_x'])
    pixel_size_y = float(rendering_params['pixel_size_y'])
    nx = int(rendering_params['nx'])
    ny = int(rendering_params['ny'])
    kernel_func_id = int(rendering_params['kernel_func_id'])
    kernel_radius = float(rendering_params['kernel_radius'])
    threads_per_chunk = rendering_params.get('threads_per_chunk', 2)
    
    # Set threads for this chunk processing
    old_nthreads = get_num_threads()
    set_num_threads(threads_per_chunk)
    
    try:
        # Load SPH grid if not provided (do this per process to avoid memory issues)
        grid_to_use = sph_grid
        if grid_to_use is None:
            try:
                if os.path.exists('/home/abhinav.roy/sph_vis/SPH_grid_512.npy'):
                    grid_to_use = np.load('/home/abhinav.roy/sph_vis/SPH_grid_512.npy')
                elif os.path.exists('/home/abhinav.roy/sph_vis/SPH_grid_100.npy'):
                    grid_to_use = np.load('/home/abhinav.roy/sph_vis/SPH_grid_100.npy')
            except Exception:
                pass  # Silently fail and continue without grid
        
        # Render chunk
        chunk_img, chunk_weight = render_projection_chunk_numba(
            chunk_pos, chunk_masses, chunk_hs, chunk_field,
            xmin, ymin, pixel_size_x, pixel_size_y,
            nx, ny, kernel_func_id, kernel_radius, grid_to_use,
        )
        
        return chunk_img, chunk_weight
    
    finally:
        # Reset thread count
        set_num_threads(old_nthreads)


@njit(nopython=True, fastmath=True, cache=True, nogil=True)
def get_kernel_func_id(kernel_name: str) -> int:
    """
    Convert kernel name to ID for use in Numba functions.
    
    Parameters
    ----------
    kernel_name : str
        Name of the kernel
        
    Returns
    -------
    int
        Kernel function ID
    """
    if kernel_name == "cubic":
        return 0
    elif kernel_name == "quintic":
        return 1
    elif kernel_name == "wendland":
        return 2
    else:
        return 0  # Default to cubic


def render_projection(
    positions: np.ndarray,
    masses: np.ndarray,
    hs: np.ndarray,
    field: Optional[np.ndarray] = None,
    plane: str = "xy",
    box: Optional[Tuple[float, float, float]] = None,
    image_shape: Tuple[int, int] = (2048, 2048),
    kernel: str = "cubic",
    kernel_radius: float = 2.0,
    method: str = "numba",
    nthreads: Optional[int] = None,
    chunk_size: int = 1_000_000,
    extent: Optional[Tuple[float, float, float, float]] = None,
    sph_grid: Optional[np.ndarray] = None,
    boundary_buffer: float = 0.05,
    parallel_chunks: int = 4,
) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
    """
    Render a 2D projection of SPH particle data.
    
    Parameters
    ----------
    positions : np.ndarray
        Particle positions, shape (N, 3)
    masses : np.ndarray
        Particle masses, shape (N,)
    hs : np.ndarray
        Smoothing lengths, shape (N,)
    field : np.ndarray, optional
        Field values to visualize, shape (N,)
    plane : str, default="xy"
        Projection plane: "xy", "xz", or "yz"
    box : Tuple[float, float, float], optional
        Box size for periodic boundary conditions
    image_shape : Tuple[int, int], default=(2048, 2048)
        Output image dimensions (ny, nx)
    kernel : str, default="cubic"
        SPH kernel: "cubic", "quintic", or "wendland"
    kernel_radius : float, default=2.0
        Multiple of smoothing length defining support radius
    method : str, default="numba"
        Computation method: only "numba" is supported
    nthreads : int, optional
        Number of threads for parallel processing
    chunk_size : int, default=2_000_000
        Maximum number of particles to process at once
    extent : Tuple[float, float, float, float], optional
        Image extent as (xmin, xmax, ymin, ymax)
    sph_grid : np.ndarray, optional
        Precomputed SPH kernel grid for faster rendering
    boundary_buffer : float, default=0.05
        Fraction of image extent to add as boundary buffer to prevent edge effects.
        Set to 0.0 to disable boundary expansion. The image is rendered with expanded
        extent and then cropped back to original size.
    parallel_chunks : int, default=4
        Number of chunks to process in parallel using concurrent futures. Set to 1 to
        disable parallel chunk processing. Higher values use more memory but can be
        faster for large datasets. Threads are divided among parallel chunks.
        
    Returns
    -------
    np.ndarray
        The rendered 2D image
    np.ndarray
        The weight map (only if field is provided)
    float, optional
        The percent error in mass conservation (only if conserve_mass is True and field is None)
    """
    # Validate inputs
    if positions.shape[0] != masses.shape[0] or positions.shape[0] != hs.shape[0]:
        raise ValueError("positions, masses, and hs must have the same length")
    
    if field is not None and positions.shape[0] != field.shape[0]:
        raise ValueError("field must have the same length as positions")
    
    # Set number of threads
    old_nthreads = get_num_threads()
    if nthreads is not None:
        set_num_threads(nthreads)
    else:
        nthreads = 10  # Default to 10 threads
        set_num_threads(nthreads)
    
    # Get projection plane indices
    plane_indices = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    if plane not in plane_indices:
        raise ValueError(f"Invalid plane: {plane}. Must be one of: {list(plane_indices.keys())}")
    
    i, j = plane_indices[plane]
    
    # Extract 2D positions for the projection
    pos_2d = np.zeros((positions.shape[0], 2), dtype=np.float32)
    pos_2d[:, 0] = positions[:, i].astype(np.float32)
    pos_2d[:, 1] = positions[:, j].astype(np.float32)
    
    # Convert inputs to float32 for efficiency
    masses = masses.astype(np.float32)
    hs = hs.astype(np.float32)
    if field is not None:
        field = field.astype(np.float32)
    
    # Determine image extent
    if extent is None:
        if box is not None:
            # Use full box extent for periodic boundary conditions
            xmin, ymin = 0.0, 0.0
            xmax, ymax = box[i], box[j]
        else:
            # Use data extent
            xmin, xmax = np.min(pos_2d[:, 0]), np.max(pos_2d[:, 0])
            ymin, ymax = np.min(pos_2d[:, 1]), np.max(pos_2d[:, 1])
            
            # Add a small margin
            margin_x = 0.05 * (xmax - xmin)
            margin_y = 0.05 * (ymax - ymin)
            xmin -= margin_x
            xmax += margin_x
            ymin -= margin_y
            ymax += margin_y
    else:
        xmin, xmax, ymin, ymax = extent
    
    # Store original extent for cropping
    orig_xmin, orig_xmax = xmin, xmax
    orig_ymin, orig_ymax = ymin, ymax
    orig_nx, orig_ny = image_shape
    
    # Add boundary buffer to prevent edge effects (if enabled)
    if boundary_buffer > 0.0:
        width_x = xmax - xmin
        width_y = ymax - ymin
        boundary_x = boundary_buffer * width_x
        boundary_y = boundary_buffer * width_y
        
        # Expand extent by boundary_buffer on all sides
        xmin_expanded = xmin - boundary_x
        xmax_expanded = xmax + boundary_x
        ymin_expanded = ymin - boundary_y
        ymax_expanded = ymax + boundary_y
        
        # Calculate expanded image dimensions to maintain pixel resolution
        pixel_size_x = width_x / orig_nx
        pixel_size_y = width_y / orig_ny
        
        # Expanded dimensions
        nx = int((xmax_expanded - xmin_expanded) / pixel_size_x)
        ny = int((ymax_expanded - ymin_expanded) / pixel_size_y)
        
        # Use expanded extent for rendering
        xmin, xmax = xmin_expanded, xmax_expanded
        ymin, ymax = ymin_expanded, ymax_expanded
    else:
        # No boundary expansion
        nx, ny = image_shape
        pixel_size_x = (xmax - xmin) / nx
        pixel_size_y = (ymax - ymin) / ny
    
    # Use Numba for SPH rendering (only supported method for now)
    if method != "numba":
        warnings.warn(f"Method '{method}' not supported. Using 'numba' for proper SPH rendering.")
        
    # Determine kernel function ID
    kernel_func_id = get_kernel_func_id(kernel)
    
    # Initialize result arrays
    result = np.zeros((ny, nx), dtype=np.float32)
    weight = np.zeros((ny, nx), dtype=np.float32)
    
    # Process particles in chunks to save memory
    n_particles = positions.shape[0]
    n_chunks = (n_particles + chunk_size - 1) // chunk_size
    
    # Prepare rendering parameters for parallel processing
    rendering_params = {
        'xmin': xmin,
        'ymin': ymin,
        'pixel_size_x': pixel_size_x,
        'pixel_size_y': pixel_size_y,
        'nx': nx,
        'ny': ny,
        'kernel_func_id': kernel_func_id,
        'kernel_radius': kernel_radius,
        'threads_per_chunk': max(1, nthreads // max(1, parallel_chunks)),
    }
    
    # Load SPH grid once if needed
    grid_to_use = sph_grid
    if grid_to_use is None:
        try:
            if os.path.exists('/home/abhinav.roy/sph_vis/SPH_grid_512.npy'):
                grid_to_use = np.load('/home/abhinav.roy/sph_vis/SPH_grid_512.npy')
            elif os.path.exists('/home/abhinav.roy/sph_vis/SPH_grid_100.npy'):
                grid_to_use = np.load('/home/abhinav.roy/sph_vis/SPH_grid_100.npy')
        except Exception as e:
            print(f"Warning: Could not load SPH grid: {e}")
    
    if parallel_chunks > 1 and n_chunks > 1:
        # Parallel chunk processing using concurrent futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_chunks) as executor:
            # Prepare chunk data
            chunk_futures = []
            
            for i in range(n_chunks):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, n_particles)
                
                chunk_pos = pos_2d[start:end]
                chunk_masses = masses[start:end]
                chunk_hs = hs[start:end]
                chunk_field = None if field is None else field[start:end]
                
                chunk_data = (chunk_pos, chunk_masses, chunk_hs, chunk_field)
                
                # Submit chunk for processing
                future = executor.submit(process_chunk_parallel, chunk_data, rendering_params, grid_to_use)
                chunk_futures.append(future)
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(chunk_futures):
                try:
                    chunk_img, chunk_weight = future.result()
                    result += chunk_img
                    weight += chunk_weight
                except Exception as e:
                    print(f"Warning: Chunk processing failed: {e}")
                    # Continue with other chunks
    else:
        # Sequential processing (fallback or when parallel_chunks=1)
        for i in range(n_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, n_particles)
            
            chunk_pos = pos_2d[start:end]
            chunk_masses = masses[start:end]
            chunk_hs = hs[start:end]
            chunk_field = None if field is None else field[start:end]
            
            # Render chunk
            chunk_img, chunk_weight = render_projection_chunk_numba(
                chunk_pos, chunk_masses, chunk_hs, chunk_field,
                float(xmin), float(ymin), float(pixel_size_x), float(pixel_size_y),
                nx, ny, kernel_func_id, kernel_radius, grid_to_use,
            )
            
            # Accumulate results
            result += chunk_img
            weight += chunk_weight
    
    # Reset number of threads to original value
    set_num_threads(old_nthreads)
    
    # Crop back to original extent to remove boundary buffer
    if boundary_buffer > 0.0 and (nx != orig_nx or ny != orig_ny):
        # Calculate crop boundaries based on current extent
        crop_left = int((orig_xmin - xmin) / pixel_size_x)
        crop_right = crop_left + orig_nx
        crop_bottom = int((orig_ymin - ymin) / pixel_size_y)
        crop_top = crop_bottom + orig_ny
        
        # Ensure crop boundaries are within image bounds
        crop_left = max(0, crop_left)
        crop_right = min(nx, crop_right)
        crop_bottom = max(0, crop_bottom)
        crop_top = min(ny, crop_top)
        
        # Crop the result and weights
        result = result[crop_bottom:crop_top, crop_left:crop_right]
        weight = weight[crop_bottom:crop_top, crop_left:crop_right]
        
        # Resize to exact original dimensions if needed
        if result.shape != (orig_ny, orig_nx):
            # Simple resize by padding or trimming
            result_cropped = np.zeros((orig_ny, orig_nx), dtype=np.float32)
            weight_cropped = np.zeros((orig_ny, orig_nx), dtype=np.float32)
            
            min_ny = min(result.shape[0], orig_ny)
            min_nx = min(result.shape[1], orig_nx)
            
            result_cropped[:min_ny, :min_nx] = result[:min_ny, :min_nx]
            weight_cropped[:min_ny, :min_nx] = weight[:min_ny, :min_nx]
            
            result = result_cropped
            weight = weight_cropped
    
    # Force mass conservation if requested
    error_percent = None
    if field is None:
        total_mass_in_image = np.sum(result)
        total_particle_mass = np.sum(masses)
        if total_mass_in_image > 0:  # Avoid division by zero
            error_percent = 100.0 * (total_mass_in_image - total_particle_mass) / total_particle_mass

    return result, weight, error_percent


def memory_estimate(
    n_particles: int,
    image_shape: Tuple[int, int] = (2048, 2048),
    n_chunks: int = 10,
    with_field: bool = True,
    precision: str = "float32"
) -> Dict[str, Union[int, str]]:
    """
    Estimate memory usage for SPH visualization.
    
    Parameters
    ----------
    n_particles : int
        Number of particles
    image_shape : Tuple[int, int], default=(2048, 2048)
        Output image dimensions (ny, nx)
    n_chunks : int, default=10
        Number of chunks to process particles
    with_field : bool, default=True
        Whether a field is provided
    precision : str, default="float32"
        Precision of calculations: "float32" or "float64"
        
    Returns
    -------
    Dict[str, Union[int, str]]
        Memory usage estimates
    """
    # Calculate bytes per element
    bytes_per_element = 4 if precision == "float32" else 8
    
    # Calculate memory usage for input data
    input_memory = n_particles * (3 + 1 + 1 + (1 if with_field else 0)) * bytes_per_element
    
    # Calculate memory usage for output image
    ny, nx = image_shape
    output_memory = ny * nx * bytes_per_element * (2 if with_field else 1)
    
    # Calculate memory usage per chunk
    chunk_size = n_particles // n_chunks
    chunk_memory = chunk_size * (3 + 1 + 1 + (1 if with_field else 0)) * bytes_per_element
    
    # Calculate total memory usage
    total_memory = input_memory + output_memory + chunk_memory

    # Calculate recommended chunk size for optimal performance
    recommended_chunk_size = max(1, chunk_size // 2)

    # Convert to human-readable format
    def human_readable_size(size_bytes):
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024**2:
            return f"{size_bytes / 1024:.2f} KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes / 1024**2:.2f} MB"
        else:
            return f"{size_bytes / 1024**3:.2f} GB"
    
    return {
        "input_memory_bytes": input_memory,
        "output_memory_bytes": output_memory,
        "chunk_memory_bytes": chunk_memory,
        "total_memory_bytes": total_memory,
        "input_memory": human_readable_size(input_memory),
        "output_memory": human_readable_size(output_memory),
        "chunk_memory": human_readable_size(chunk_memory),
        "total_memory": human_readable_size(total_memory),
        "recommended_chunk_size": recommended_chunk_size,
        "precision": precision,
    }


# ROTATE POSITIONS ALONG ANY AXIS CENTERED AT ANY POINT (X, Y, Z)
def rotate_positions(positions, axis, angle, center):
	"""
	Rotate positions around a specified axis by a given angle centered at a specific point.

	Parameters:
	-----------
	positions : np.ndarray
		Array of shape (N, 3) containing the x, y, z coordinates of N particles.
	axis : str
		Axis to rotate around ('x', 'y', or 'z').
	angle : float
		Angle in degrees to rotate.
	center : tuple
		Tuple of (x_center, y_center, z_center) specifying the center of rotation.

	Returns:
	--------
	rotated_positions : np.ndarray
		Array of shape (N, 3) containing the rotated x, y, z coordinates.
	"""
	# Convert angle from degrees to radians
	theta = np.radians(angle)
	cos_theta = np.cos(theta)
	sin_theta = np.sin(theta)

	# Translate positions to origin based on center
	translated_positions = positions - np.array(center)

	# Initialize rotation matrix
	if axis == 'x':
		rotation_matrix = np.array([[1, 0, 0],
									[0, cos_theta, -sin_theta],
									[0, sin_theta, cos_theta]])
	elif axis == 'y':
		rotation_matrix = np.array([[cos_theta, 0, sin_theta],
									[0, 1, 0],
									[-sin_theta, 0, cos_theta]])
	elif axis == 'z':
		rotation_matrix = np.array([[cos_theta, -sin_theta, 0],
									[sin_theta, cos_theta, 0],
									[0, 0, 1]])
	else:
		raise ValueError("Axis must be 'x', 'y', or 'z'.")
	
	# Rotate positions
	rotated_positions = np.dot(translated_positions, rotation_matrix.T)

	# Translate positions back to original center
	rotated_positions += np.array(center)

	return rotated_positions    
    
    


def save_frame(image: np.ndarray, save_path: str, frame_idx: int, format: str = "npy") -> None:
    """
    Save a single frame image to disk.
    
    Parameters
    ----------
    image : np.ndarray
        2D image array
    save_path : str
        Directory path to save frames
    frame_idx : int
        Frame index for filename
    format : str, default="png"
        Image format
    """
    import os
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # save it as a npy binary file
    filename = os.path.join(save_path, f"frame_{frame_idx:04d}.{format}")
    np.save(filename, image)


def render_rotation(
    positions: np.ndarray,
    masses: np.ndarray,
    hs: np.ndarray,
    field: Optional[np.ndarray] = None,
    plane: str = "xy",
    axis: str = "y",
    n_frames: int = 360,
    extent: Optional[Tuple[float, float, float, float]] = None,
    center: Optional[Tuple[float, float, float]] = None,
    box: Optional[Tuple[float, float, float]] = None,
    image_shape: Tuple[int, int] = (1024, 1024),
    kernel: str = "cubic",
    kernel_radius: float = 2.0,
    method: str = "numba",
    nthreads: Optional[int] = None,
    chunk_size: int = 1_000_000,
    save_path: Optional[str] = None,
    return_stack: bool = False,
    angle_range: Tuple[float, float] = (0.0, 360.0),
    show_progress: bool = True,
    parallel_chunks: int = 4,
    sph_grid: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    Render rotating SPH projections of a particle dataset.
    
    This function creates smooth 3D rotation animations by applying rotation matrices
    to particle positions before projection. Memory efficiency is maintained by
    processing particles in chunks and applying rotations per chunk rather than
    storing all rotated positions.
    
    Parameters
    ----------
    positions : np.ndarray
        Particle positions, shape (N, 3)
    masses : np.ndarray
        Particle masses, shape (N,)
    hs : np.ndarray
        Smoothing lengths, shape (N,)
    field : np.ndarray, optional
        Field values to visualize, shape (N,)
    plane : str, default="xy"
        Projection plane: "xy", "xz", or "yz"
    axis : str, default="y"
        Rotation axis: "x", "y", or "z"
    n_frames : int, default=360
        Number of frames to render
    extent : Tuple[float, float, float, float], optional
        Image extent as (xmin, xmax, ymin, ymax). If None, auto-calculated
    center : Tuple[float, float, float], optional
        Center of rotation. If None, uses dataset center
    box : Tuple[float, float, float], optional
        Box size for periodic boundary conditions
    image_shape : Tuple[int, int], default=(1024, 1024)
        Output image dimensions (ny, nx)
    kernel : str, default="cubic"
        SPH kernel: "cubic", "quintic", or "wendland"
    kernel_radius : float, default=2.0
        Multiple of smoothing length defining support radius
    method : str, default="numba"
        Computation method: only "numba" is supported
    nthreads : int, optional
        Number of threads for parallel processing
    chunk_size : int, default=1_000_000
        Maximum number of particles to process at once
    save_path : str, optional
        Directory path to save individual frame images
    return_stack : bool, default=False
        If True, return all frames as np.ndarray of shape (n_frames, H, W)
    angle_range : Tuple[float, float], default=(0.0, 360.0)
        Range of rotation angles in degrees (start, end)
    show_progress : bool, default=True
        Whether to show progress bar
    parallel_chunks : int, default=4
        Number of chunks to process in parallel for each frame
    sph_grid : np.ndarray, optional
        Precomputed SPH kernel grid for faster rendering
        
    Returns
    -------
    np.ndarray, optional
        If return_stack=True, returns array of shape (n_frames, H, W)
        Otherwise returns None
        
    Examples
    --------
    >>> # Render 180-frame rotation and save as individual images
    >>> render_rotation(positions, masses, hs, 
    ...                axis="y", n_frames=180, 
    ...                save_path="frames")
    
    >>> # Render full rotation and return as stack
    >>> frames = render_rotation(positions, masses, hs,
    ...                         axis="y", n_frames=360,
    ...                         return_stack=True)
    """
    # Validate inputs
    if positions.shape[0] != masses.shape[0] or positions.shape[0] != hs.shape[0]:
        raise ValueError("positions, masses, and hs must have the same length")
    
    if field is not None and positions.shape[0] != field.shape[0]:
        raise ValueError("field must have the same length as positions")
    
    # Calculate the extent if not provided
    projection_extent = extent
    if projection_extent is None:
        if box is not None:
            # Use full box extent for periodic boundary conditions
            projection_extent = (0.0, box[0], 0.0, box[1])
        else:
            # Use data extent
            xmin, xmax = np.min(positions[:, 0]), np.max(positions[:, 0])
            ymin, ymax = np.min(positions[:, 1]), np.max(positions[:, 1])
            
            # Add a small margin
            margin_x = 0.05 * (xmax - xmin)
            margin_y = 0.05 * (ymax - ymin)
            projection_extent = (xmin - margin_x, xmax + margin_x, ymin - margin_y, ymax + margin_y)
    
    # Determine center of rotation if not provided
    if center is None:
        if box is not None:
            center = np.array([box[0] / 2.0, box[1] / 2.0, box[2] / 2.0])
        else:
            center = np.array([np.mean(positions[:, 0]), np.mean(positions[:, 1]), np.mean(positions[:, 2])])
        

    # generate the angles to be used for rotation
    angles = np.linspace(angle_range[0], angle_range[1], n_frames, endpoint=False)
    
    # Initialize results
    frame_stack = None
    if return_stack:
        ny, nx = image_shape
        frame_stack = np.zeros((n_frames, ny, nx), dtype=np.float32)

    print(f"Rendering {n_frames} frames rotating around '{axis}' axis...")

    # print info about memory usage
    mem_info = memory_estimate(
        n_particles=positions.shape[0],
        image_shape=image_shape,
        n_chunks=(positions.shape[0] + chunk_size - 1) // chunk_size,
        with_field=(field is not None),
        precision="float32"
    )
    print("Estimated memory usage:")
    for key, value in mem_info.items():
        print(f"  {key}: {value}")
    
    # print info about chunk processing and total particles 
    print(f"Processing {positions.shape[0]} particles in chunks of {chunk_size} with {parallel_chunks} parallel chunks and total {nthreads} threads per frame.")
    print(f"Image shape: {image_shape}, Kernel: {kernel} (radius={kernel_radius}), Method: {method}")
    print("Angle range: ", angle_range)
    print("Angles (degrees): ", angles)
    
    # Setup progress bar
    iterator = range(n_frames)
    if show_progress and tqdm is not None:
        iterator = tqdm(iterator, desc="Rendering frames")
    
    # Render each frame
    for frame_idx in iterator:

        rotated_positions = rotate_positions(
            positions=positions,
            axis=axis,
            angle=angles[frame_idx],
            center=center
        )


        # Render the frame
        frame_image, _, _ = render_projection(
            positions=rotated_positions,
            masses=masses,
            hs=hs,
            field=field,
            plane=plane,
            box=box,
            image_shape=image_shape,
            kernel=kernel,
            kernel_radius=kernel_radius,
            method=method,
            nthreads=nthreads,
            chunk_size=chunk_size,
            extent=projection_extent,
            sph_grid=sph_grid,
            parallel_chunks=parallel_chunks,
        )

        # Store in stack if requested
        if return_stack:
            frame_stack[frame_idx] = frame_image
        
        # Save frame if path provided
        if save_path is not None:
            save_frame(frame_image, save_path, frame_idx, format="npy")
            

    return frame_stack if return_stack else None