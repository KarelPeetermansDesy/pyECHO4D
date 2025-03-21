import numpy as np
import matplotlib.pyplot as plt
from  pyLib4ECHO import *

# Input directories and filenames
#N3_RectangularStepOut_LongQuadWake
#mydir = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N3_RectangularStepOut_LongQuadWake/Results/'
#N5_PohangDechirper
#mydir = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N5_PohangDechirper/Results/'
#N6_TaperedCollimator
#mydir = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N6_TaperedCollimator/Results/'
#N7_TeslaCavitywithCouplers
#mydir = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N7_TeslaCavitywithCouplers/Results/'
#N8_FlatTaperWithFieldMonitor
#mydir = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N8_FlatTaperWithFieldMonitor/Results/'
#N12_RoundCorrugatedShifted
#mydir = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N12_RoundCorrugatedShifted/Results/'
#N13_BNL_Flange_LongQuadWake
#mydir = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N13_BNL_Flange_LongQuadWake/Results/'
#N14_KEK_Collimator
#mydir = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N14_KEK_Collimator/Results/'
#N15_BNL_Bellows
mydir = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N15_BNL_Bellows/Results/'



InputFile = f'{mydir}input.txt'
InWakeFileY = f'{mydir}wake3DindirectY.bin'
InWakeFileZ = f'{mydir}wake3DindirectZ.bin'
OutFileY = 'wakeQy.dat'
OutFileZ = 'wakeQz.dat'
di = 1

# Read input parameters
nx, ny, nz, hx, hy, hz, sigma, iy0, iz0, BCy, BCz, GFile, Ymin, Zmin= ReadInput(InputFile)

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

if iy0 > 1:
    wake[:, 1] = (W[iz0-1, iy0-1, :] - W[iz0-1, iy0-1 + di - 2, :]) / (hy * di)
else:
    wake[:, 1] = W[iz0-1, iy0-1 + di - 1, :] / (hy * (di - 0.5))

wake[:, 1] *= 1e3  # Convert from 1/mm to 1/m

# Fill the first column with longitudinal positions
for i in range(nx):
    wake[i, 0] = -5 * sigma + hx * (i + 1)

np.savetxt(f'{mydir}{OutFileY}', wake, delimiter=' ', fmt='%.6e')

KickQy = LongLoss3d(hx, -wake[:, 1], sigma)
print('KickQy = ', KickQy);

# Plot and save the wake potential in Y direction
plt.subplot(2, 1, 1)
plt.plot(wake[:, 0], wake[:, 1])
plt.title('"Quadrupole" Wake Potential, KickQy = {:.4e} V/pC/m'.format(KickQy[0]))
plt.ylabel('W_y [V/pC/m]')

# Process wake in Z direction
W0, nz1, ny1, nx1, z1, y1 = ReadDataB(InWakeFileZ)
W = Vect2Matr3D(W0, nx1, ny1, nz1)
W0 = []

if iz0 > 1:
    wake[:, 1] = (W[iz0-1, iy0-1,:] - W[iz0 -1 + di - 2, iy0-1, :]) / (hz * di)
else:
    wake[:, 1] = W[iz0 -1 + di - 1, iy0 -1, :] / (hz * (di - 0.5))

wake[:, 1] *= 1e3  # Convert from 1/mm to 1/m

np.savetxt(f'{mydir}{OutFileZ}', wake, delimiter=' ', fmt='%.6e')

KickQz = LongLoss3d(hx, -wake[:, 1], sigma)
print('KickQz = ', KickQz);
# Plot and save the wake potential in Z direction
plt.subplot(2, 1, 2)
plt.plot(wake[:, 0], wake[:, 1])
plt.title('"Quadrupole" Wake Potential, KickQz = {:.4e} V/pC/m'.format(KickQz[0]))
plt.xlabel('s [mm]')
plt.ylabel('W_z [V/pC/m]')
plt.tight_layout()

plt.show()
