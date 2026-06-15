import os
import numpy as np
from classy import Class

def gen_powerspectrum(redshift,filepath:str=None):
    # STEP - 1 : Define your cosmological parameters
    # - Create a params dictionary (Equivalent of .ini file).
    # - What is not specified will be set to CLASS default parameters.
    # - Do check the difference between small and capital initials (Omega_m vs omega_m=Omega_m*(h**2))
    #   hint: look for value of omega_cdm in explanatory.ini
    params = {
        'output':'mPk',
        'non linear':'halofit',
        'Omega_b':0.0493,
        'Omega_m':0.3153,
        'h':0.6736,
        'A_s':2.1e-9,
        'n_s':0.9649,
        'P_k_max_h/Mpc':1e3,
        'z_max_pk':99,
    }

    # STEP 2 : Create an instance of the CLASS wrapper.
    cosmo = Class()

    # STEP 3 : # STEP 3 : Set the parameters to the cosmological code.
    cosmo.set(params)

    # STEP 4 : Run the whole code.
    # - Depending on your output, it will call the CLASS modules.
    # - Without any output asked, CLASS will only compute background quantities like a(t),H(z) etc.
    cosmo.compute()

    # STEP 5 : Extract the power spectrum at the desired redshift.
    # Reference for units: classy.pyx -> def pk_lin(...)
    h=params['h']
    k = np.logspace(-5, 3, 1000)    # k in h/Mpc
    kh = k*h                        # k in 1/Mpc

    Pnonlin = np.array([cosmo.pk(ki, redshift) for ki in kh])
    Plin = np.array([cosmo.pk_lin(ki, redshift) for ki in kh])

    # Convert to (Mpc/h)^3
    Plin *= h**3
    Pnonlin *= h**3

    # STEP 6 : Save the power spectrum to a file if filepath is provided.
    if filepath is not None:
        np.savetxt(filepath, np.column_stack((k, Plin, Pnonlin)), header='k [h/Mpc]  P_lin [(Mpc/h)^3]  P_nonlin [(Mpc/h)^3]', fmt='%.6e')

    return k, Plin, Pnonlin


def get_powerspectrum(redshift, cache_dir:str):
    filename = f"classy_pk_z{redshift}.txt"
    filepath = f"{cache_dir}/{filename}"

    if not os.path.exists(filepath):
        print(f"Power spectrum not found in cache. Generating and saving to: {filepath}")
        gen_powerspectrum(redshift, filepath)    
        
    print(f"Loading power spectrum from cache: {filepath}")
    k, pk_lin, pk_nonlin = np.loadtxt(filepath, unpack=True)
    
    return k, pk_lin, pk_nonlin



if __name__ == "__main__":
    REDSHIFT = 0
    CACHE_DIR = 'tools'#"/home/ranit/gfs/cache"
    k, pk_lin, pk_nonlin = get_powerspectrum(REDSHIFT, CACHE_DIR)

    # Example: Plotting the power spectrum
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.loglog(k, pk_lin, label='Linear Power Spectrum')
    plt.loglog(k, pk_nonlin, label='Non-linear Power Spectrum')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('P(k) [(Mpc/h)$^3$]')
    plt.title(f'Power Spectrum at z={REDSHIFT}')
    plt.legend(frameon=False, loc='lower left')
    plt.show()

    # Compare with MP-Gadget tools