import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'notebook', 'no-latex'])
from scipy.special import wofz

# unit_conversion_paramters
kms_to_cms = 1e5
mpc_to_cm = 3.086e24
ang_to_cm = 1e-8
# physical_constants
c = 2.99792458e10
kB = 1.380658e-16
mp = 1.6726e-24
me = 9.1094e-28
mn = 1.6749286e-24
# cosmology
omega_m = 0.308
omega_l  = 0.692
omega_r  = 0.0
omega_k = 0.0
omega_b = 0.0482
hubble_param = 0.678
sigma_8 = 0.829
Y  = 0.24
ns = 0.961

def atomic_constants(inp):
    # for Ly alpha
    lambda_r = 1215.6701
    f12 = 0.416
    damp_gamma = 6.265e8
    ialpha = 4.469e-18
    npp = 1
    ne = 1
    nn = 0
    inp.update(vars())
    return

def main(inp):
    atomic_constants(inp)

    npixel = 2048
    nlos = 1
    z_arr   = np.zeros((npixel,nlos))
    vXI_arr   = z_arr.copy()
    nXI_arr = z_arr.copy() + 1e-20
    TXI_arr = z_arr.copy() + 1e-9
    tauXI_arr = z_arr.copy()

    nXI_arr[1024,:] = 1e-9 # corresponds to typical overdensity of 1
    TXI_arr[1024,:] = 1e4

    # everything is in cm/s
    vXI_arr[1024,:] = 0 * kms_to_cms 
    bXI_arr = np.sqrt(2*kB*TXI_arr/(mp+me)) 
    
    inp.update(vars())
    generate_spectrum(inp)
    return

def calculate_profile(z_val,z_arr,vXI_arr_1d,bXI_arr_1d,nXI_arr_1d, lambda_in_cm, damp_gamma, c):
    # note: everything here is in CGS (cm/s)
    velocity_factor_array = (vXI_arr_1d + (c*(z_arr-z_val)/(1+z_val)))/bXI_arr_1d
    alpha_factor_arr = (damp_gamma*lambda_in_cm)/(4*np.pi*bXI_arr_1d)
    z_complex_arr = velocity_factor_array + 1j*alpha_factor_arr
    voigt_calc = wofz(z_complex_arr).real
    
    tau_val_pixel = np.sum(nXI_arr_1d*voigt_calc/(bXI_arr_1d*(1.0+z_arr)), axis=0)

    return tau_val_pixel

def generate_spectrum(inp):
    TXI_arr = inp['TXI_arr']
    bXI_arr = inp['bXI_arr']
    tauXI_arr = inp['tauXI_arr']
    nXI_arr, vXI_arr, nlos, npixel = inp['nXI_arr'], inp['vXI_arr'], inp['nlos'], inp['npixel']
    print(bXI_arr[1024, :])

    lambda_in_cm, damp_gamma = inp['lambda_r']*ang_to_cm, inp['damp_gamma']

    ialpha = inp['ialpha']
    dl_box_cm = 40*mpc_to_cm/npixel/hubble_param
    z_arr = np.linspace(2.97,3.03,2048) 

    for los_idx in range(nlos):
        nXI_arr_1d = nXI_arr[:,los_idx]
        bXI_arr_1d = bXI_arr[:,los_idx]
        vXI_arr_1d = vXI_arr[:,los_idx]

        for pixel_idx in range(npixel):
            z_val = z_arr[pixel_idx]
            tauXI_arr_val = calculate_profile(z_val,z_arr,vXI_arr_1d,bXI_arr_1d,nXI_arr_1d,lambda_in_cm, damp_gamma, c)
            tauXI_arr[pixel_idx, los_idx] = tauXI_arr_val
            # print(nXI_arr_1d, vXI_arr_1d, bXI_arr_1d)
            # exit()

    tauXI_arr = tauXI_arr*c*ialpha*dl_box_cm/np.sqrt(np.pi)

    inp.update({'tauXI_arr': tauXI_arr})
    print(np.max(tauXI_arr))

    # now plot
    fig, ax = plt.subplots(nrows=5,sharex=True, figsize=(8,12))
    ax[0].step(z_arr, np.log10(inp['nXI_arr']))
    ax[1].step(z_arr, np.log10(inp['TXI_arr']))
    ax[2].step(z_arr, inp['vXI_arr'] / kms_to_cms) 
    ax[3].step(z_arr, tauXI_arr)
    ax[4].step(z_arr, np.exp(-inp['tauXI_arr']))
    
    ax[0].set_ylabel(r'$\log [n_\mathrm{HI}$ (cm$^{-3}$)]')
    ax[1].set_ylabel(r'$\log [T$ (K)]')
    ax[2].set_ylabel(r'$v$ [km s$^{-1}$]')
    ax[3].set_ylabel(r'$\tau$')
    ax[4].set_ylabel(r'$F = e^{-\tau}$')
    ax[4].set_xlabel(r'$z$')
    ax[4].set_xlim(2.97,3.03)

    plt.tight_layout()
    plt.savefig('one_line.png')
    plt.close()
    return

inp = {}
main(inp)