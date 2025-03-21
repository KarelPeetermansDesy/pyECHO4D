import numpy as np
import matplotlib.pyplot as plt
from  pyLib4ECHO import *

# Input directories and filenames
mydir0 = '-'
#N2_RoundStepOut_LongWake_PEC
#mydir1 = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N2_RoundStepOut_DipoleWake_PEC/Results/'
#N2_RoundStepOut_LongWake_Vacuum
#mydir1 = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N2_RoundStepOut_DipoleWake_Vacuum/Results/'
#N4_RectangularStepOut_DipoleWake
#mydir1 = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N4_RectangularStepOut_DipoleWake/Results/'
#N13_BNL_Flange_DipoleWake
#mydir1 = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N13_BNL_Flange_DipoleWake/Results/'

OutFileY = 'wakeDy.dat'

# If mydir0 is '-', handle the condition
if mydir0 == '-':
    iy0 = 1
else:
    # Read the input and data for the first directory
    nx, ny, nz, hx, hy, hz, sigma, iy0, iz0, BCy, BCz, GFile, Ymin, Zmin= ReadInput(f'{mydir0}/input.txt')
    W0, nz1, ny1, nx1, z1, y1 = ReadDataB(f'{mydir0}/wake3DindirectY.bin')
    W = Vect2Matr3D(W0, nx1, ny1, nz1)
    W0 = []

    wake1 = np.zeros((nx, 2))
    if iy0 > 1:
        iy0 = 2
    if iz0 > 1:
        iz0 = 2

    if iy0 > 1:
        wake1[:, 1] = (W[iz0-1, iy0-1, :] + W[iz0-1, iy0-1 - 1, :]) * 0.5
    else:
        wake1[:, 1] = W[iz0-1, iy0-1, :] * 0

# Read the input and data for the second directory
nx, ny, nz, hx, hy, hz, sigma, iy1, iz0, BCy, BCz, GFile, Ymin, Zmin= ReadInput(f'{mydir1}/input.txt')
W0, nz1, ny1, nx1, z1, y1 = ReadDataB(f'{mydir1}/wake3DindirectY.bin')
W = Vect2Matr3D(W0, nx1, ny1, nz1)
W0 = []

wake2 = np.zeros((nx, 2))

if iy1 > 1:
    iy1 = 2
if iz0 > 1:
    iz0 = 2

iy0 = iy0
iy1 = iy1
iz0 = iz0


# Populate the wake2 array
for i in range(nx):
    wake2[i, 0] = -5 * sigma + hx * (i + 1)

dy = hy
if mydir0 == '-':
    wake2[:, 1] = 0.5 * W[iz0-1, iy1-1, :] / dy
else:
    wake2[:, 1] = (W[iz0-1, iy1-1, :] + W[iz0-1, iy1-1 - 1, :]) * 0.5
    wake2[:, 1] = (wake2[:, 1] - wake1[:, 1]) / dy

wake2[:, 1] = wake2[:, 1] * 1e3  # Convert from 1/mm to 1/m

# Save the data to a file
np.savetxt(f'{mydir1}/{OutFileY}', wake2, delimiter=' ', fmt='%.6e')

# Calculate the LongLoss3d
KickDy = LongLoss3d(hx, -wake2[:, 1], sigma)
print ('KickDy = ',KickDy)

# Plot the result
plt.plot(wake2[:, 0], wake2[:, 1])
plt.title('"Dipole" Wake Potential, KickDy = {:.3e}'.format(KickDy[0]))
plt.xlabel('s [mm]')
plt.ylabel('W_y [V/pC/m]')
plt.show()