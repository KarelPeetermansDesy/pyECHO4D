import matplotlib.pyplot as plt
from  pyLib4ECHO import *

#N1_RoundStepOut_LongWake_PEC
#InputFile = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N1_RoundStepOut_LongWake_PEC/input.txt'
#fname = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N1_RoundStepOut_LongWake_PEC/Fields/Ey.bin'
#N2_RoundStepOut_LongWake_Vacuum
#InputFile = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N1_RoundStepOut_LongWake_Vacuum/input.txt'
#fname = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N1_RoundStepOut_LongWake_Vacuum/Fields/Ey.bin'
#N2_RoundStepOut_DipoleWake_PEC
#InputFile = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N2_RoundStepOut_DipoleWake_PEC/input.txt'
#fname = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N2_RoundStepOut_DipoleWake_PEC/Fields/Ey.bin'
#N7_TeslaCavitywithCouplers
InputFile = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N7_TeslaCavitywithCouplers/input.txt'
fname = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N7_TeslaCavitywithCouplers/Fields/Ey.bin'

# Load the binary data for Ey field
W, nz, ny, length, z, y = ReadDataB(fname)
# Convert the field data into a 3D matrix
Ey = Vect2Matr3D(W, length, ny, nz)

# Read input file

nx, ny, nz, hx, hy, hz, sigma, iy0, iz0, BCy, BCz, GFile, Ymin, Zmin= ReadInput(InputFile)

# Initial conditions
n0=int(np.floor(sigma / hx * 5))-1;
posy = iy0 -1
posz = iz0 -1

# Create subplots (with a 3D subplot for the first plot)
fig, axs = plt.subplots(2, 1, figsize=(8, 10))

# Create a 3D axis for the first subplot
ax1 = fig.add_subplot(2, 1, 1, projection='3d')

# Subplot 1: Mesh plot for Ey (3D plot)
X, Y = np.meshgrid(y, z)
ax1.plot_surface(X, Y, Ey[:, :, n0], cmap='viridis')
ax1.set_title('Ey at s=0 cm')
ax1.set_xlabel('z[cm]')
ax1.set_ylabel('y[cm]')
ax1.set_zlabel('Ey[a.u.]')

# Subplot 2: Comparison to free space solution
hy = y[1] - y[0]
#axs[1].plot(y, Ey[posz, :, n0] / hy * (sigma * np.sqrt(2 * np.pi)))
axs[1].plot(y, Ey[posz, :, n0] / hy *2* (sigma * np.sqrt(2 * np.pi)))

ys = y - (posy - 0.5) * hy - np.min(y)
Eya = 20. / ys
axs[1].plot(y, Eya, 'k')
axs[1].set_title('Comparison to the free space solution (in black)')
axs[1].set_xlabel('y[cm]')
axs[1].set_ylabel('Ey[a.u.]')

plt.tight_layout()
plt.show()

# Plot the Ey field along a specific slice
I = Ey[posz, posy, :]
plt.figure()
plt.plot(I)
plt.show()