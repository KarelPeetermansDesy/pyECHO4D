from geom import *

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from scipy.constants import c

from ECHO2D_Functions import * 

from scipy.special import jv, yv
import cmath
from scipy import optimize as opt
import time
from WaveguideFunctions import *

import matplotlib.pyplot as plt
import numpy as np
from Plotting import *
from scipy.optimize import brentq

# Define the function and parameters
def generate_logarithmic_coefficients_exciting(R0, R1, L, k=25.0):
    a = (R1 - R0) / np.log(k * L + 1)
    b = k
    c = R0
    return [a, b, c]

def logarithmic(z, a, b, c):
    return a * np.log(b * z + 1) + c

def log_taper(z, R0, R1, L, k):
    """
    Normalized logarithmic taper between R0 and R1 over length L.
    - k<1: slow start, fast finish
    - k>1: fast start, slow finish
    - k->0: linear
    """
    # avoid division by zero when k is extremely small
    if abs(k) < 1e-8:
        return R0 + (R1 - R0) * (z / L)
    return R0 + (R1 - R0) * np.log1p(k * z / L) / np.log1p(k)

# Constants from user's script
c = 299792458
gamma = 32000
vPhase = (1 - 1/gamma**2)**(1/2) * c
er = 3.81
thickness = 60e-6
Omega0 = 2*np.pi*500e9
Omega1 = 2*np.pi*1500e9
length = 0.1
nSamples = 20000
RadiusSamples = np.linspace(20e-6, 6e-3, nSamples)

# Dummy D0x function for testing purposes
# Replace this with the actual imported D0x function from your codebase


# Calculate Radius0 and Radius1
D0x_store_0 = D0x(Omega0, Omega0/vPhase, RadiusSamples, RadiusSamples + thickness, er)
D0x_store_1 = D0x(Omega1, Omega1/vPhase, RadiusSamples, RadiusSamples + thickness, er)

try:
    FirstZeroCrossing_0 = np.where(np.diff(np.sign(D0x_store_0)))[0][0]
except:
    FirstZeroCrossing_0 = 0

try:
    FirstZeroCrossing_1 = np.where(np.diff(np.sign(D0x_store_1)))[0][0]
except:
    FirstZeroCrossing_1 = 0

nChirp = 1000
nChirp = 500
nSamples = 20000
length = 10e-2
zChirp = np.linspace(0, length, nChirp)
# Omega1 = 2*np.pi*500e9
# Omega0 = 2*np.pi*1500e9 
OmegaChirp = (Omega0 - Omega1) * zChirp/length + Omega1

Radius1 = RadiusSamples[FirstZeroCrossing_0]
Radius0 = RadiusSamples[FirstZeroCrossing_1]

kzStart = kr1(Omega0, Omega0/vPhase, er)
kzEnd = kr1(Omega1, Omega1/vPhase, er)


# Recompute radius profiles using updated Radius0 and Radius1
def linear_taper(z, R0, R1, L, slope):
    """
    Simple linear taper.
    slope=0   -> constant radius = R0
    slope=1   -> full ramp from R0 to R1 over [0,L]
    0 < slope < 1 -> partial ramp (final R0 + slope*(R1-R0))
    """
    return R0 + slope*(R1 - R0)*(z / L)

# choose how many different slopes you want    
Ntotal = 30
slopes = np.linspace(0, 1, Ntotal)  # from flat (0) up to full (1)

# prepare storage for CSV
radius_data = {}

plt.figure(figsize=(6, 4), dpi=150)
for i, s in enumerate(slopes):
    Rz = linear_taper(zChirp, Radius0, Radius1, length, s)
    radius_data[f'slope={s:.2f}'] = Rz
    
    plt.plot(
        zChirp * 1e2,       # cm
        Rz * 1e6,           # µm
        color=plt.cm.viridis(i / (len(slopes)-1)),
        label=f's={s:.2f}'
    )

plt.xlabel('z (cm)')
plt.ylabel('Radius (µm)')
plt.title('Linear Taper Profiles (varying slope)')
plt.grid(True)
sm = plt.cm.ScalarMappable(
    cmap='viridis',
    norm=plt.Normalize(vmin=0, vmax=1)
)
sm.set_array([])
plt.colorbar(sm, label='slope')
plt.tight_layout()

# write out
outdir = 'output'
os.makedirs(outdir, exist_ok=True)
plt.savefig(os.path.join(outdir, 'linear_taper_profiles.png'))
plt.close()

df = pd.DataFrame(radius_data, index=zChirp)
df.index.name = 'z (m)'
df.to_csv(os.path.join(outdir, 'linear_taper_profiles.csv'))
