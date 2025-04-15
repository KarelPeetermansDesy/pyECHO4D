#!/usr/bin/env python3
import sys
import os

# Check for the proper number of command-line arguments
if len(sys.argv) != 6:
    print("Usage: python3 ASTRA2ECHO_geom.py <a_val> <delta_val> <eps_val> <dlw_length> <remaining>")
    sys.exit(1)

# Get parameters from command-line and convert to float
a_val      = float(sys.argv[1])*1e-1    # This replaces 0.2 in the file.
delta_val  = float(sys.argv[2])*1e-4
eps_val    = float(sys.argv[3])    # This replaces 3.81 in the file.
dlw_length = float(sys.argv[4])*1e2
remaining  = float(sys.argv[5])*1e-1    # This replaces 8.0 (the tube start)

# Compute derived values
a_plus_delta   = a_val + delta_val            # Replaces 0.22 where needed
rem_plus_len   = remaining + dlw_length         # Replaces 153.0 in the file
metal_len      = rem_plus_len + 5                     # This is the length of the metal part

# Now, create the content for pipe.txt.
# Numbers are formatted to two decimals (adjust as needed) and values are inserted in place of the
# original hard-coded numbers.
lines = [
    "% Number of materials",
    "2",
    "% Number of elements in metal with conductive walls, permeability, permitivity, conductivity",
    "5 1 1 0",
    "% Segments of lines and elipses with conductivity",
    # In the following segments, the third field (tube start) becomes remaining.
    f"-0.1\t1\t{remaining:.2f}\t1\t0\t0\t0\t0\t1\t0.01",
    # Both the first and third fields represent the tube start here.
    f"{remaining:.2f}\t1\t{remaining:.2f}\t{a_plus_delta:.6f}\t0\t0\t0\t0\t1\t0.01",
    # In this segment, the first field is remaining, the second field is a_plus_delta, and the third field is remaining+dlw_length.
    f"{remaining:.2f}\t{a_plus_delta:.6f}\t{metal_len:.2f}\t{a_plus_delta:.6f}\t0\t0\t0\t0\t1\t0",
    # Here, the tube end (first and third fields) become remaining+dlw_length.
    f"{metal_len:.2f}\t{a_plus_delta:.6f}\t{metal_len:.2f}\t1\t0\t0\t0\t0\t1\t0",
    # This segment leaves the third field as '150' unchanged.
    f"{metal_len:.2f}\t1\t150\t1\t0\t0\t0\t0\t1\t0",
    "% Number of elements in material 1, permetivity, permeability, conductivity",
    # The epsilon value replaces 3.81.
    f"4 {eps_val:.2f} 1 0",
    "% Segments of lines and elipses",
    # In the following segments of material 1, the first and third fields originally “8.0” become remaining;
    # the second field 0.2 becomes a_val; the fourth field 0.22 becomes a_plus_delta.
    f"{remaining:.2f}\t{a_val:.4f}\t{remaining:.2f}\t{a_plus_delta:.6f}\t0\t0\t0\t0\t1\t0",
    f"{remaining:.2f}\t{a_plus_delta:.6f}\t{rem_plus_len:.2f}\t{a_plus_delta:.6f}\t0\t0\t0\t0\t1\t0",
    # In this line, the first and third fields are remaining+dlw_length and the fourth field is a_val.
    f"{rem_plus_len:.2f}\t{a_plus_delta:.6f}\t{rem_plus_len:.2f}\t{a_val:.4f}\t0\t0\t0\t0\t1\t0",
    # Finally, here the first field is remaining+dlw_length, the second and fourth fields are a_val, and the third field is remaining.
    f"{rem_plus_len:.2f}\t{a_val:.4f}\t{remaining:.2f}\t{a_val:.4f}\t0\t0\t0\t0\t1\t0"
]

# Write the content to pipe.txt
with open("pipe.txt", "w") as f:
    for line in lines:
        f.write(line + "\n")

print("pipe.txt created successfully.")
