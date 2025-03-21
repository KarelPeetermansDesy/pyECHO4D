import numpy as np
from  pyLib4ECHO import *

# ---------------- INPUT ----------------
#N2_RoundStepOut_LongWake_PEC
#mydir = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N2_RoundStepOut_DipoleWake_PEC/Results/'
#N2_RoundStepOut_LongWake_Vacuum
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
#N12_RoundCorrugatedShifted
#mydir = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N12_RoundCorrugatedShifted/Results/'
#N13_BNL_Flange_DipoleWake
#mydir = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N13_BNL_Flange_DipoleWake/Results/'
#N13_BNL_Flange_LongQuadWake
#mydir = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N13_BNL_Flange_LongQuadWake/Results/'
#N14_KEK_Collimator
#mydir = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N14_KEK_Collimator/Results/'
#N15_BNL_Bellows
mydir = '/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N15_BNL_Bellows/Results/'



InputFile = mydir + 'input.txt'
InWakeFile = 'wake3Dindirect.bin'
OutWakeFileY = 'wake3DindirectY.bin'
OutWakeFileZ = 'wake3DindirectZ.bin'

# ---------------- READ INPUT FILE ----------------
nx, ny, nz, hx, hy, hz, sigma, iy0, iz0, BCy, BCz, GFile, Ymin, Zmin = ReadInput(InputFile)

# ---------------- READ WAKE DATA ----------------
W0, nz1, ny1, nx1, z1, y1 = ReadDataB(mydir + InWakeFile)
W=Vect2Matr3D(W0,nx1,ny1,nz1)
W0 = []
nz, ny, nx = W.shape

# ---------------- PROCESS WAKE DATA ----------------
Wy = np.zeros((nz, ny - 1, nx))
y = y1[:ny - 1] + 0.5 * hy
Wz = np.zeros((nz - 1, ny, nx))
z = z1[:nz - 1] + 0.5 * hz
WI = np.zeros((nz, ny, nx))

for j in range(ny):
    for k in range(nz):
        WI[k, j, :] = IntegrTr(hx, W[k, j, :])

Wy[:, :ny - 1, :] = (WI[:, :ny - 1, :] - WI[:, 1:ny, :]) / hy
Wz[:nz - 1, :, :] = (WI[:nz - 1, :, :] - WI[1:nz, :, :]) / hz

# ---------------- COMPUTE KICK FACTOR ----------------
KickY = np.zeros((ny - 1, nz))
for i in range(ny - 1):
    for j in range(nz):
        br = Wy[j, i, :]
        KickY[i, j] = LongLoss3d(hx, br, sigma)[0]

# ---------------- SAVE OUTPUT ----------------
# File paths
fname = mydir + OutWakeFileY
W0 = Matr2Vect3D(Wy, nx1, ny1-1, nz1)
WriteDataB(fname, W0, nz1, ny1-1, nx1, z1, y)

fname = mydir + OutWakeFileZ
W0 = Matr2Vect3D(Wz, nx1, ny1, nz1-1)
WriteDataB(fname, W0, nz1-1, ny1, nx1, z, y1)