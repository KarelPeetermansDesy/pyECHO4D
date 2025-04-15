#!/bin/bash
# Define the ASTRA folder (change this to your folder's full path)
ASTRA_FOLDER="/home/karelp/wakefields/AREAL Optimisation/ASTRA/FALCO Deck/BatchTemp/astra_219.33"

# parse_rfgun_in.sh
# This script reads the file rfgun.in, extracts:
#   1. The DLW inner radius (Ap_R(1)) and computes the DLW length as Ap_Z2(1) - Ap_Z1(1)
#      from the &APERTURE section.
#   2. The wake parameters (a, delta, epsilon) from the wake filename in the &WAKE section.
#
# It then saves these values to a file called "parsed_info.txt".

# Input file (assumed to be in the same folder as this script)
input_file="rfgun.in"

if [ ! -f "$ASTRA_FOLDER/$input_file" ]; then
    echo "Input file $ASTRA_FOLDER/$input_file not found."
    exit 1
fi

# Extract the &APERTURE block (from the line starting with &APERTURE until the next line with just /)
aperture=$(awk '/&APERTURE/,/^\// {print}' "$ASTRA_FOLDER/$input_file")

# Extract the required parameters from the APERTURE block.
# We assume the file has lines like "Ap_R(1)=2.0", "Ap_Z1(1)=1.8," and "Ap_Z2(1)=5.05,"
inner_radius=$(echo "$aperture" | grep -i "Ap_R(1)=" | sed "s/.*Ap_R(1)=\([^, ]*\).*/\1/")
ap_z1=$(echo "$aperture" | grep -i "Ap_Z1(1)=" | sed "s/.*Ap_Z1(1)=\([^, ]*\).*/\1/")
ap_z2=$(echo "$aperture" | grep -i "Ap_Z2(1)=" | sed "s/.*Ap_Z2(1)=\([^, ]*\).*/\1/")

# Compute the DLW length as Ap_Z2 - Ap_Z1 using awk for floating point arithmetic.
dlw_length=$(awk -v z1="$ap_z1" -v z2="$ap_z2" 'BEGIN {printf "%.3f", z2 - z1}')

# Now extract the wake filename. We pick the first occurrence of a line with "Wk_filename"
wake_line=$(grep -i "Wk_filename" "$ASTRA_FOLDER/$input_file" | head -n 1)
# Extract the filename between single quotes
wake_filename=$(echo "$wake_line" | sed "s/.*'\(.*\)'.*/\1/")

# From the wake filename extract a, delta, and epsilon.
# Expected wake filename format, for example:
# '../../../../ECHO_wakes/w_a2.00mm_delta360.66um_eps3.81_gamma12.31_Nmode10.dat'
# This extracts:
#   - a:      number after "w_a" and before "mm"
#   - delta:  number between "delta" and "um"
#   - epsilon: number between "eps" and "_gamma"
a_val=$(echo "$wake_filename" | sed -n "s/.*w_a\([0-9.]*\)mm.*/\1/p")
delta_val=$(echo "$wake_filename" | sed -n "s/.*_delta\([0-9.]*\)um.*/\1/p")
eps_val=$(echo "$wake_filename" | sed -n "s/.*_eps\([0-9.]*\)_gamma.*/\1/p")

# Save the extracted information into an output file.
output_file="parsed_info.txt"
{
    echo "DLW inner radius: $inner_radius"
    echo "DLW length: $dlw_length"
    echo "Wake parameter a: $a_val"
    echo "Wake parameter delta: $delta_val"
    echo "Wake parameter epsilon: $eps_val"
} > "$output_file"

echo "Parsed information saved to $output_file."




# Define the starting DLW coordinate in meters
START_DLW="$ap_z1"

# Convert START_DLW (meters) to millimeters as an integer.
START_MM=$(echo "$START_DLW" | awk '{printf "%.0f", $1 * 1000}')


# Initialize variables for tracking the best match
bestCoord=-1
bestFile=""

# Enable nullglob so that the pattern expands to nothing if no matches are found
shopt -s nullglob

# Loop through the files with the expected pattern
for file in "$ASTRA_FOLDER"/rfgun.*.*; do
    base=$(basename "$file")
    # Extract the coordinate portion from the filename.
    # Expected format: rfgun.<coordinate>.<other>
    coord=$(echo "$base" | cut -d '.' -f2)
    
    # Convert the coordinate to a number (treating it as base10 to remove any leading zeros)
    coord_val=$((10#$coord))
    
    # Check if the coordinate is strictly less than the target coordinate (in mm)
    if (( coord_val < START_MM )); then
        # And check if it is closer than any previously found candidate
        if (( coord_val > bestCoord )); then
            bestCoord=$coord_val
            bestFile="$base"
        fi
    fi
done

# If no suitable file was found, inform the user and exit.
if [ -z "$bestFile" ]; then
    echo "No suitable file found in $ASTRA_FOLDER."
    exit 1
fi

# Calculate the remaining distance in millimeters
remaining=$(( START_MM - bestCoord ))

echo "Best matching file: $bestFile"
echo "Remaining distance to DLW entrance (mm): $remaining"

# Run the Python script with the found filename and remaining distance as arguments.
ECHO_File="${bestFile/%001/.ast}"
python3 ASTRA2ECHO.py "$ASTRA_FOLDER/$bestFile" "$ECHO_File"

python3 ASTRA2ECHO_geom.py "$a_val" "$delta_val" "$eps_val" "$dlw_length" "$remaining"
