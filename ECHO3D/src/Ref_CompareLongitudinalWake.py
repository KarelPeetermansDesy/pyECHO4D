import numpy as np
import matplotlib.pyplot as plt

#N1_RoundStepOut_LongWake_PEC
#file1='/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N1_RoundStepOut_LongWake_PEC/Reference/wake.dat'
#file2='/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N1_RoundStepOut_LongWake_PEC/Results/wakeL.dat'
#N1_RoundStepOut_LongWake_PEC
#file1='/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N1_RoundStepOut_LongWake_Vacuum/Reference/wake.dat'
#file2='/Users/zagor/DESY/ECHO/ACurrentRelease/ECHO3D_v1_5/Examples/N1_RoundStepOut_LongWake_Vacuum/Results/wakeL.dat'

# Load the data files
w1 = np.genfromtxt(file1, comments='%')
w1 = w1[1:]  # Remove the first row (similar to w1(1,:) in MATLAB)
w2 = np.genfromtxt(file2, comments='%')

# Create the plot
plt.figure()

# Plot the data
plt.plot(w1[:, 0] * 10, w1[:, 1], label='2D', linewidth=2)  # Multiply the first column by 10 (convert to mm)
plt.plot(w2[:, 0], w2[:, 1], label='3D', linewidth=2)

# Set plot title and labels
plt.title('Comparison 2D (in blue) with 3D')
plt.xlabel('s[mm]')
plt.ylabel('W_|_|[V/pC]')

# Set x-axis limits
plt.xlim([-5, 20])

# Show the legend
plt.legend()

# Display the plot
plt.show()
