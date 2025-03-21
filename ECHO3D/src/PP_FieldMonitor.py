import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from  pyLib4ECHO import *
from scipy.interpolate import interp2d


# INPUT
mydir = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N8_FlatTaperWithFieldMonitor/Results/'
FieldFile = mydir + 'Monitor_N01.txt'
PM_x = 0.03
PM_y = 0.005

# Read field file
with open(FieldFile, 'r') as ff:
    line = ff.readline().strip()
    Field = line.split('Field=')[1].split()[0]
    timetype = line.split('time=')[1].strip()

# Body
if timetype == 's':
    T, X, Y, Z, F, kt, kx, ky, kz = ReadFieldMonitor_stime(FieldFile)
    lab = 'z [m]'
    MeshPos = np.zeros(kt)
else:
    T, S, Y, Z, F, kt, ks, ky, kz = ReadFieldMonitor_xtime(FieldFile)
    lab = 's [m]'
    MeshPos = F[:, 0]
    X = -S
    kx = ks

F = F[:, 1:] # Removing the first column of F
PM_Field = np.zeros(kt)

# Initialize your plot
fig = plt.figure()

# Create a 3D subplot
ax = fig.add_subplot(211, projection='3d')
# Create a second subplot for PM_Field vs T
ax2 = fig.add_subplot(212)

# Initialize the line and point plot to be updated
surface = None
point = None

for i in range(kt):
    FF = np.reshape(F[i, :ky * kx], (ky, kx))  # Reshape F to kx by ky and transpose it

    # Create the 2D interpolation function
    interp_func = interp2d(MeshPos[i] + X, Y, FF, kind='linear', fill_value=0)

    # Perform the interpolation for the given PM_x and PM_y
    PM_Field[i] = interp_func(PM_x, PM_y)

    # Clear the previous surface and point from the plot
    ax.cla()  # Clear the axes (this removes the previous surface plot)

    # 3D Plotting: Create the meshgrid for the X and Y axes
    X_mesh, Y_mesh = np.meshgrid(X, Y)

    # Plot the surface in 3D (redraw surface every iteration)
    surface = ax.plot_surface(X_mesh, Y_mesh, FF, cmap='viridis', rstride=1, cstride=1, alpha=0.8)
    point = ax.scatter(PM_x - MeshPos[i], PM_y, 0, color='k', marker='.', s=100)  # Add point for PM_x, PM_y

    # Set labels and title
    ax.set_xlabel(lab)
    ax.set_ylabel('y [m]')
    ax.set_zlabel(f'{Field}/Q [V/m/nC]')
    ax.set_title(f'Field = {Field} type={timetype} ct = {T[i]} m')

    # Clear and update the second subplot (PM_Field vs T)
    ax2.cla()  # Clear the second subplot (PM_Field vs T)
    ax2.plot(T, PM_Field)  # Redraw the plot for PM_Field vs T
    ax2.set_title(f'x = {PM_x}; y = {PM_y}')
    ax2.set_xlabel('ct [m]')
    ax2.set_ylabel(f'{Field}/Q [V/m/nC]')

    # Pause to update the plot
    plt.pause(0.01)
    plt.tight_layout()

plt.show()  # Show the plot at the end

# Output
out = np.column_stack((T, PM_Field))
np.savetxt(mydir + 'PointMonitor.txt', out, fmt='%f')

# Constants
eps0 = 8.854187817000000e-12
V = 2 / PM_y * 1 / np.sqrt(2 * np.pi) / 0.005 * 1 / (4 * np.pi * eps0) * 1e-9

print(f'Calculated Value V = {V:.6e}')
