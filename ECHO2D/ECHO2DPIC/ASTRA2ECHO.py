import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# File paths (using raw strings to avoid escaping backslashes)
# Use the final definitions as in your MATLAB code.
astrafile = sys.argv[1]
echofile = sys.argv[2]

# Scaling factor
factor = 1

# Load the ASCII data from the Astra file.
# (If you have a custom load function that does more than np.loadtxt, please let me know.)
F1 = np.loadtxt(astrafile)

# Sub-sample the data using the factor.
F2 = F1[::factor, :].copy()

# Multiply column 8 (MATLAB's 8th column, index 7 in Python) by the factor.
F2[:, 7] = F2[:, 7] * factor

# Adjust the first row, 7th column (MATLAB: F2(1,7); Python: F2[0,6])
F2[0, 6] = F2[0, 6] - np.max(F2[:, 6])

# Starting position in meters
z0 = -0.001
# Calculate the shift: maximum of rows 2:end in column 3 (MATLAB: F2(2:end,3))
shift = np.max(F2[1:, 2])
# Adjust first row, column 3 (MATLAB: F2(1,3); Python: F2[0,2])
F2[0, 2] = z0 - shift

# Create a directory for figures if it doesn't exist
figs_folder = "figs"
os.makedirs(figs_folder, exist_ok=True)

# Create the figure with two subplots (vertical stacking)
fig, axs = plt.subplots(2, 1, figsize=(8, 10))

# First subplot: Histogram of (F2(2:end,3) + F2(1,3))
# MATLAB indexing F2(2:end,3) -> Python: F2[1:, 2]; then add F2(1,3) -> F2[0,2]
positions = F2[1:, 2] + F2[0, 2]
axs[0].hist(positions, bins=200)
axs[0].set_title("Position")
axs[0].set_xlabel("z[m]")

# Second subplot: Histogram of F2(:,7) (MATLAB's 7th column -> Python: column index 6)
times = F2[:, 6]
axs[1].hist(times, bins=200)
axs[1].set_title("Time")
axs[1].set_xlabel("ct[m]")

plt.tight_layout()
# Save the figure to the figs folder
plt.savefig(os.path.join(figs_folder, "particles_ASTRA2ECHO.png"))
# (Do not call plt.show() as requested)

# Calculate the total charge, Q, as the sum over column 8 (MATLAB: F2(:,8) -> Python: F2[:,7])
Q = np.sum(F2[:, 7])
print("Q =", Q)

# Save F2 as an ASCII file to echofile
np.savetxt(echofile, F2, fmt="%g", delimiter="\t")
