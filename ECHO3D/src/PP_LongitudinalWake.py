import matplotlib.pyplot as plt
from  pyLib4ECHO import *

# Input files and directories
in_wake_file = 'wake3Dindirect.bin'
out_file_name = 'wakeL.dat'

#N1_RoundStepOut_LongWake_PEC
#mydir = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N1_RoundStepOut_LongWake_PEC/Results/'
#N1_RoundStepOut_LongWake_Vacuum
#mydir = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N1_RoundStepOut_LongWake_Vacuum/Results/'
#N2_RoundStepOut_DipoleWake_PEC
#mydir = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N2_RoundStepOut_DipoleWake_PEC/Results/'
#N2_RoundStepOut_DipoleWake_Vacuum
#mydir = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N2_RoundStepOut_DipoleWake_Vacuum/Results/'
#N3_RectangularStepOut_LongQuadWake
#mydir = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N3_RectangularStepOut_LongQuadWake/Results/'
#N4_RectangularStepOut_DipoleWake
#mydir = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N4_RectangularStepOut_DipoleWake/Results/'
#N5_PohangDechirper
#mydir = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N5_PohangDechirper/Results/'
#N6_TaperedCollimator
#mydir = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N6_TaperedCollimator/Results/'
#N7_TeslaCavitywithCouplers
#mydir = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N7_TeslaCavitywithCouplers/Results/'
#N8_FlatTaperWithFieldMonitor
#mydir = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N8_FlatTaperWithFieldMonitor/Results/'
#N9_Round_Dielectric
#mydir = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N9_Round_Dielectric/ECHO3D/Results/'
#N10_Round_Ferromagnetic
#mydir = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N10_Round_Ferromagnetic/ECHO3D/Results/'
#N11_BPM
#mydir = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N11_BPM/Results/'
#N12_RoundCorrugatedShifted
#mydir = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N12_RoundCorrugatedShifted/Results/'
#N13_BNL_Flange_LongQuadWake
#mydir = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N13_BNL_Flange_LongQuadWake/Results/'
#N14_KEK_Collimator
#mydir = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N14_KEK_Collimator/Results/'
#N15_BNL_Bellows
#mydir = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N15_BNL_Bellows/Results/'
#N16_FreeSpaceIris
#mydir = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N16_FreeSpaceIris/ECHO3D/Results/'
#in_wake_file = 'wake3D.bin'
#N17_CeramicBreak
mydir = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N17_CeramicBreak/Results/'

input_file = mydir + 'input.txt'

dy, dz = 0, 0  # Offset in meshlines

# Read input data
nx, ny, nz, hx, hy, hz, sigma, iy0, iz0, BCy, BCz, GFile, Ymin, Zmin = ReadInput(input_file)

if iy0 > 1:
    iy0 = 2
if iz0 > 1:
    iz0 = 2

ypos = iy0 + dy - 1
zpos = iz0 + dz - 1
outdir = mydir
h = hx  # Longitudinal step in cm

# Read wake data
fname = mydir + in_wake_file
W0, nz1, ny1, length1, z1, y1 = ReadDataB(fname)
FW = Vect2Matr3D(W0, length1, ny1, nz1)

nz, ny, nx = FW.shape
br = FW[zpos, ypos, :]

x_arg = np.array([-5 * sigma + h * i for i in range(1,nx+1)])
LossL = LongLoss3d(h, br, sigma)
print(f'Loss={LossL} [V/pC]')

wake = np.zeros((nx, 2))
wake[:, 0] = x_arg
wake[:, 1] = br

LossM = np.zeros((ny, nz))
for i in range(ny):
    for j in range(nz):
        br = FW[j, i, :]
        LossM[i, j] = LongLoss3d(h, br, sigma)[0]

# Plot results
fig = plt.figure(figsize=(8, 10))

# Create the first subplot with 3D projection
axs1 = fig.add_subplot(211, projection='3d')  # 3D axis for the first subplot

# Create a meshgrid for 3D plotting
X, Y = np.meshgrid(z1, y1)
z_min, z_max = np.min(LossM), np.max(LossM)  # Get the min and max values of LossM
axs1.set_zlim(z_min*0.99, z_max*1.01)

# Create a surface plot
surf = axs1.plot_surface(X, Y, LossM, cmap='viridis')

# Set the title and labels for the first subplot
axs1.set_title('Loss Factor')
axs1.set_xlabel('z[cm]')
axs1.set_ylabel('y[cm]')
axs1.set_zlabel('Loss [V/pC]')

# Create the second subplot (2D axis by default)
axs2 = fig.add_subplot(212)  # 2D axis for the second subplot

# Plot the longitudinal profile of wake potential
axs2.plot(wake[:, 0], wake[:, 1])

# Set title and labels for the second subplot
axs2.set_title('Longitudinal Profile of Wake Potential, LossL = {:.4e} V/pC'.format(LossL[0]))
axs2.set_xlabel('s[mm]')
axs2.set_ylabel('W [V/pC]')

# Adjust the layout to avoid overlap
plt.tight_layout()

# Show the plot
plt.show()
# Show the plot
plt.show()