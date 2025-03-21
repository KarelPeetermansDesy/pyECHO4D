import numpy as np
import matplotlib.pyplot as plt
from  pyLib4ECHO import *

# Input directories and filenames
#N7_TeslaCavitywithCouplers
#mydir = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N7_TeslaCavitywithCouplers/Results/'
#N12_RoundCorrugatedShifted
mydir = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N12_RoundCorrugatedShifted/Results/'

InputFile = f'{mydir}input.txt'
InWakeFileY = f'{mydir}wake3DindirectY.bin'
InWakeFileZ = f'{mydir}wake3DindirectZ.bin'
OutFileY = 'wakeMy.dat'
OutFileZ = 'wakeMz.dat'
di = 1

# Read input parameters
nx, ny, nz, hx, hy, hz, sigma, iy0, iz0, BCy, BCz, GFile, Ymin, Zmin = ReadInput(InputFile)

# Adjust indices
if iy0 > 1:
    iy0 = 2
if iz0 > 1:
    iz0 = 2

# Process wake in Y direction
W0, nz1, ny1, nx1, z1, y1 = ReadDataB(InWakeFileY)
W = Vect2Matr3D(W0, nx1, ny1, nz1)
W0 = []

wake = np.zeros((nx, 2))
wake[:, 1] = 0.5 * (W[iz0-1, iy0-1, :] + W[iz0-1, iy0-1 - 1, :])

# Fill the first column with longitudinal positions
for i in range(nx):
    wake[i, 0] = -5 * sigma + hx * (i + 1)

np.savetxt(f'{mydir}{OutFileY}', wake, delimiter=' ', fmt='%.6e')

KickY = LongLoss3d(hx, -wake[:, 1], sigma)
print('KickY = ', KickY);

# Plot and save the wake potential in Y direction
plt.subplot(2, 1, 1)
plt.plot(wake[:, 0], wake[:, 1])
plt.title('"Monopole" Wake Potential, KickY = {:.4e} V/pC'.format(KickY[0]))
plt.xlabel('s [cm]')
plt.ylabel('W_y [V/pC]')

# Process wake in Z direction
W0, nz1, ny1, nx1, z1, y1 = ReadDataB(InWakeFileZ)
W = Vect2Matr3D(W0, nx1, ny1, nz1)
W0 = []

wake[:, 1] = 0.5 * (W[iz0-1, iy0-1, :] + W[iz0-1 - 1, iy0-1, :])

np.savetxt(f'{mydir}{OutFileZ}', wake, delimiter=' ', fmt='%.6e')

KickZ = LongLoss3d(hx, -wake[:, 1], sigma)
print('KickZ = ', KickZ);

# Plot and save the wake potential in Z direction
plt.subplot(2, 1, 2)
plt.plot(wake[:, 0], wake[:, 1])
plt.title('"Monopole" Wake Potential, KickZ = {:.4e} V/pC'.format(KickZ[0]))
plt.xlabel('s [mm]')
plt.ylabel('W_z [V/pC]')

plt.tight_layout()
plt.show()
