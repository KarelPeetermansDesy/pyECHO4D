import numpy as np
import struct
import re
import ast

def ReadInput(InputFile):
    # Open the input file and read all lines
    with open(InputFile, 'r') as fid:
        lines = fid.readlines()

    # Initialize a dictionary to store the variables
    variables = {}

    # Process each line from the input file
    for line in lines:
        line = line.strip()
        if not line or line.startswith('%'):  # Skip empty lines or comments
            continue

        # Remove inline comments after '%'
        line = line.split('%')[0].strip()

        # Match and process 'VariableName = value'
        match = re.match(r"(\w+)\s*=\s*(.*)", line)
        if match:
            var_name, var_value = match.groups()
            var_value = var_value.strip()

            # Convert MATLAB-style array to Python list (e.g., [1 2 3] -> [1, 2, 3])
            if re.match(r"\[.*\]", var_value):
                var_value = re.sub(r'\s+', ',', var_value)  # Replace spaces with commas inside brackets

            try:
                variables[var_name] = ast.literal_eval(var_value)  # Safe evaluation
            except:
                variables[var_name] = var_value  # Store as string if evaluation fails

    # Extract values safely
    nx = variables.get('MeshLength')
    hx, hy, hz = variables.get('Steps', [None, None, None])
    dY = variables.get('dY', [0, 0])
    dZ = variables.get('dZ', [0, 0])
    BunchPosition = variables.get('BunchPosition', [0, 0])

    ny = round((dY[1] - dY[0]) / hy) + 1 if hy else None
    nz = round((dZ[1] - dZ[0]) / hz) + 1 if hz else None
    iy0 = BunchPosition[0] + 1
    iz0 = BunchPosition[1] + 1
    BCy = variables.get('BoundaryConditionsY')
    BCz = variables.get('BoundaryConditionsZ')
    GFile = variables.get('GeometryFile')
    sigma = variables.get('BunchSigma')
    Ymin = dY[0]
    Zmin = dZ[0]

    return nx, ny, nz, hx, hy, hz, sigma, iy0, iz0, BCy, BCz, GFile, Ymin, Zmin

# Function to read binary wake data from a file
def ReadDataB(fname):
    with open(fname, 'rb') as f:
        nx, ny, nz = np.fromfile(f, dtype=np.int32, count=3)
        y = np.fromfile(f, dtype=np.float64, count=ny)
        z = np.fromfile(f, dtype=np.float64, count=nz)
        W = np.fromfile(f, dtype=np.float64, count=nx * ny * nz)

    return W, nz, ny, nx, z, y


# Function to convert a vector into a 3D matrix
def Vect2Matr3D(V, nx, ny, nz):
    A = np.zeros((nz, ny, nx))
    index = 0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                A[iz, iy, ix] = V[index]
                index += 1
    return A


# Function to calculate longitudinal loss and spread
def LongLoss3d(h, w, sigma):
    n = len(w)
    x = np.array([-5 * sigma + h * i for i in range(1,n+1)])
    bi2 = gauss(x, sigma)
    loss = -np.sum(bi2 * w * h)
    spread = np.sqrt(np.sum(bi2 * (w + loss) ** 2 * h))

    return loss, spread


# Gaussian function
def gauss(x, sigma):
    return np.exp(-x ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))


def IntegrTr(h, x):
    n = len(x)
    y = np.zeros(n)
    y[0] = 0
    for j in range(1, n):  # Starting from index 1 (equivalent to MATLAB index 2)
        y[j] = y[j - 1] + 0.5 * (x[j] + x[j - 1])
    y = y * h

    return y


def Vect2Matr3D(V, nx, ny, nz):
    A = np.zeros((nz, ny, nx))  # Initialize the output matrix
    nynz = ny * nz

    for iz in range(nz):  # Python uses 0-based indexing
        for iy in range(ny):
            for ix in range(nx):
                ind = ix * nynz + iy * nz + iz  # Calculate the index
                A[iz, iy, ix] = V[ind]  # Assign the value to the 3D array

    return A


def Matr2Vect3D(A, nx, ny, nz):
    # Flatten the 3D matrix A into a 1D vector V
    V = np.zeros(nx * ny * nz)

    nynz = ny * nz  # Calculate the product of ny and nz for the indexing

    # Convert the 3D matrix A into a 1D vector V
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                # Calculate the index for the 1D vector
                ind = ix * nynz + iy * nz + iz
                V[ind] = A[iz, iy, ix]  # Assign the value from the 3D matrix to the 1D vector

    return V


def WriteDataB(fname, W, nz, ny, nx, z, y):
    with open(fname, 'wb') as fid:
        # Write the dimensions (nx, ny, nz) as int32
        fid.write(struct.pack('3i', nx, ny, nz))  # Writing three int32 values

        # Write the y and z arrays as double (float64)
        fid.write(struct.pack(f'{ny}d', *y))  # Writing y as a sequence of doubles
        fid.write(struct.pack(f'{nz}d', *z))  # Writing z as a sequence of doubles

        # Write the W array as double (float64)
        W_flat = W.flatten()  # Flatten W to 1D for writing
        fid.write(struct.pack(f'{len(W_flat)}d', *W_flat))  # Writing W as a sequence of doubles

def ReadFieldMonitor_xtime(field_file):
    with open(field_file, 'r') as f:
        lines = f.readlines()

    # Read metadata
    line=lines[0].strip()
    field = line.split('Field=')[1].split()[0]
    timetype = line.split('time=')[1].strip()

    line = lines[1].lstrip('%').strip()
    kt = float(line.split('k_ct=')[1].split()[0])
    ht = float(line.split('h_ct=')[1].split()[0])
    t0 = float(line.split('ct0=')[1].split()[0])
    line = lines[2].lstrip('%').strip()
    ky = float(line.split('k_y=')[1].split()[0])
    hy = float(line.split('h_y=')[1].split()[0])
    y0 = float(line.split('y0=')[1].split()[0])
    line = lines[3].lstrip('%').strip()
    kz = float(line.split('k_z=')[1].split()[0])
    hz = float(line.split('h_z=')[1].split()[0])
    z0 = float(line.split('z0=')[1].split()[0])
    line = lines[4].lstrip('%').strip()
    ks = float(line.split('k_s=')[1].split()[0])
    hs = float(line.split('h_s=')[1].split()[0])
    s0 = float(line.split('s0=')[1].split()[0])

    # Load the numerical data (assuming space-separated values)
    data = np.loadtxt(field_file, skiprows=5)

    # Generate coordinate arrays
    T = np.array([t0 + ht * i for i in range(int(kt))])
    Y = np.array([y0 + hy * i for i in range(int(ky))])
    Z = np.array([z0 + hz * i for i in range(int(kz))])
    S = np.array([s0 + hs * i for i in range(int(ks))])

    return T, S, Y, Z, data, int(kt), int(ks), int(ky), int(kz)


def ReadFieldMonitor_stime(field_file):
    with open(field_file, 'r') as f:
        lines = f.readlines()

    # Read metadata
    line=lines[0].strip()
    field = line.split('Field=')[1].split()[0]
    timetype = line.split('time=')[1].strip()

    line = lines[1].lstrip('%').strip()
    kt = float(line.split('k_ct=')[1].split()[0])
    ht = float(line.split('h_ct=')[1].split()[0])
    t0 = float(line.split('ct0=')[1].split()[0])
    line = lines[2].lstrip('%').strip()
    ky = float(line.split('k_y=')[1].split()[0])
    hy = float(line.split('h_y=')[1].split()[0])
    y0 = float(line.split('y0=')[1].split()[0])
    line = lines[3].lstrip('%').strip()
    kz = float(line.split('k_z=')[1].split()[0])
    hz = float(line.split('h_z=')[1].split()[0])
    z0 = float(line.split('z0=')[1].split()[0])
    line = lines[4].lstrip('%').strip()
    kx = float(line.split('k_x=')[1].split()[0])
    hx = float(line.split('h_x=')[1].split()[0])
    x0 = float(line.split('x0=')[1].split()[0])

    # Load the numerical data (assuming space-separated values)
    data = np.loadtxt(field_file, skiprows=5)

    # Generate coordinate arrays
    T = np.array([t0 + ht * i for i in range(int(kt))])
    Y = np.array([y0 + hy * i for i in range(int(ky))])
    Z = np.array([z0 + hz * i for i in range(int(kz))])
    X = np.array([x0 + hx * i for i in range(int(kx))])

    return T, X, Y, Z, data, int(kt), int(kx), int(ky), int(kz)

