import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat


# Input directory
#N1_RoundStepOut_LongWake_PEC
#MeshDir = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N1_RoundStepOut_LongWake_PEC/Mesh/body_00'
#N1_RoundStepOut_LongWake_Vacuum
#MeshDir = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N1_RoundStepOut_LongWake_Vacuum/Mesh/body_00'
#N2_RoundStepOut_DipoleWake_PEC
#MeshDir = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N2_RoundStepOut_DipoleWake_PEC/Mesh/body_00'
#N7_TeslaCavitywithCouplers
MeshDir= '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N7_TeslaCavitywithCouplers/Mesh/DownCoupler_00'


# Function to read binary data (equivalent to MATLAB fread)
def read_binary(filename):
    with open(filename, 'rb') as fid:
        N = np.fromfile(fid, dtype=np.int32, count=3)
        nx, ny, nz = map(int, N)  # Convert to Python integers
        np_size = nx * ny * nz
        W1 = np.fromfile(fid, dtype=np.float32, count=3 * np_size)
    return nx, ny, nz, W1

# Load Eps.mat data
nx, ny, nz, W1 = read_binary(f'{MeshDir}/Eps.mat')
np_size = nx * ny * nz

Lx = np.zeros((nz, ny, nx), dtype=np.float32)
Ly = np.zeros((nz, ny, nx), dtype=np.float32)
Lz = np.zeros((nz, ny, nx), dtype=np.float32)

for ix in range(nx):
    for iy in range(ny):
        for iz in range(nz):
            ind = iz * nx * ny + iy * nx + ix
            Lx[iz, iy, ix] = W1[ind]
            Ly[iz, iy, ix] = W1[ind + np_size]
            Lz[iz, iy, ix] = W1[ind + 2 * np_size]

# Load Mue.mat data
nx, ny, nz, W1 = read_binary(f'{MeshDir}/Mue.mat')
np_size = nx * ny * nz

Sx = np.zeros((nz, ny, nx), dtype=np.float32)
Sy = np.zeros((nz, ny, nx), dtype=np.float32)
Sz = np.zeros((nz, ny, nx), dtype=np.float32)

for ix in range(nx):
    for iy in range(ny):
        for iz in range(nz):
            ind = iz * nx * ny + iy * nx + ix
            Sx[iz, iy, ix] = W1[ind]
            Sy[iz, iy, ix] = W1[ind + np_size]
            Sz[iz, iy, ix] = W1[ind + 2 * np_size]

# Boundary adjustments
Sy[:, 0, :] *= 0.5
Lx[:, 0, :] *= 0.5
Lz[:, 0, :] *= 0.5
Sy[:, ny-1, :] *= 0.5
Lx[:, ny-1, :] *= 0.5
Lz[:, ny-1, :] *= 0.5

Sz[0, :, :] *= 0.5
Lx[0, :, :] *= 0.5
Ly[0, :, :] *= 0.5
Sz[nz-1, :, :] *= 0.5
Lx[nz-1, :, :] *= 0.5
Ly[nz-1, :, :] *= 0.5

# Create figure with 6 subplots for 3D surfaces
fig = plt.figure(figsize=(12, 8))
ax_list = [fig.add_subplot(3, 2, i+1, projection='3d') for i in range(6)]
plt.ion()  # Enable interactive mode
index = 0  # Track current slice

# Create X, Y meshgrid for 3D surface plots
X, Y = np.meshgrid(np.arange(ny), np.arange(nz))

# Improved colormap and shading
cmap_choice = 'plasma'  # Can also try 'plasma', 'magma', 'cividis'

def update_plots():
    """ Update the plots with the current slice index """
    fig.suptitle(f'Slice {index + 1} of {nx}', fontsize=14, fontweight='bold')  # Show current slice number

    data_list = [Lx[:, :, index], Ly[:, :, index], Lz[:, :, index],
                 Sx[:, :, index], Sy[:, :, index], Sz[:, :, index]]
    titles = ['Lx', 'Ly', 'Lz', 'Sx', 'Sy', 'Sz']

    for ax, data, title in zip(ax_list, data_list, titles):
        ax.cla()  # Clear the previous plot
        ax.plot_surface(X, Y, data, cmap=cmap_choice, antialiased=True)
        ax.set_title(title)

    plt.draw()
    plt.pause(0.01)

def on_key(event):
    """ Handle key press to update plots """
    global index
    index = (index + 1) % nx  # Move to next slice, looping at nx
    update_plots()

# Connect key press event
fig.canvas.mpl_connect('key_press_event', on_key)

# Initial plot
update_plots()

# Keep the program running
plt.show(block=True)
