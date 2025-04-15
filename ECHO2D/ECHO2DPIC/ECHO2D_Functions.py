import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
from scipy.constants import c, mu_0, epsilon_0
import os
from scipy.optimize import curve_fit
import cmasher as cmr
cmap = cmr.wildfire
from matplotlib.colors import LogNorm
from matplotlib.colors import SymLogNorm
# cmap = plt.get_cmap('cmr.wild')
import re
from matplotlib.colors import LinearSegmentedColormap, Normalize
DesyColors = ['#009FDF','#f18f4d']
# DesyBlueCmap = LinearSegmentedColormap.from_list("custom_blue_white", ["white",DesyColors[0]])
DesyBlueCmap = LinearSegmentedColormap.from_list("custom_green_black_red", [DesyColors[0],"black",DesyColors[1]])
cmap=DesyBlueCmap


def parse_Iz(DAT_DIR,p,ax=None, ax2=None):
    # for index in range(2, 3):
    dim = p["k_r"] + 1
    # df = pd.read_csv(f"{DAT_DIR}/Iz0.txt", skiprows=0, header=0, names=range(dim + 1), sep=" ")
    df = pd.read_csv(f"{DAT_DIR}/Iz0.txt", skiprows=0, header=0, sep="\t")

    # df = df.iloc[:(len(df) - skip_rows_end)]  # Select the first nrows rows

    # df.drop(dim, axis=1, inplace=True)

    s = []
    # E = []
    Iz0 = []

    for i, row in df.iterrows():
        # if i < :
        # s.append(float(row.name[0]))
        # print(row.name)
        s.append(float(row[0]))
        # s.append(float(row.name))
        # E.append(np.average(row[1:dim]))

        # Iz0.append((row[1:dim]))
        # if i == 4:
        # print(row[1:dim])
        Iz0.append([float(x) for x in row[1:dim]])
    Iz0 = np.array(Iz0)
    s = np.array(s)

    print(s)
    print(np.shape(Iz0))
    # print((Iz0[4,:]))
    # print(Iz0[5:-1,18])
    # print(Iz0[5:-1,19])
    # print(Iz0[5:-1,20])
    # print(np.shape(s))
    # print(np.any(Iz0 > 0, axis=0))

    # print(np.where(np.any(Iz0 > 0, axis=0)))


    column_without_zeros = np.where(np.any(Iz0 > 0, axis=0))[0][0]
    print(column_without_zeros)
    Iz0Column = Iz0[:, column_without_zeros]
    if ax is not None:
        plt.sca(ax)
        plt.plot(s*1e6, Iz0Column, label='Current profile')
        plt.xlabel('s (µm)')
        plt.ylabel(f'Iz0 column {column_without_zeros}')

        # sigmaI = 50e-6
        # maxIndex = np.argmax(Iz0Column)
        # plt.axvline(x=(s[maxIndex]-sigmaI)*1e6, color='r', linestyle='--',alpha=0.6, label='1 sigma')
        # plt.axvline(x=(s[maxIndex]+sigmaI)*1e6, color='r', linestyle='--',alpha=0.6, label='_nolegend')   
        # plt.xlim([(s[maxIndex]-7*sigmaI)*1e6, (s[maxIndex]+7*sigmaI)*1e6])
        # plt.savefig(f'{DIR}/Iz0_column_{column_without_zeros}.png', dpi=300, bbox_inches='tight')

    if ax2 is not None:
        plt.sca(ax2)

        # Calculate the Fourier transform of the Iz0 column
        t = s / c
        Iz0Column_fft = np.fft.fftshift(np.fft.fft(Iz0Column))
        frequencies = np.fft.fftshift(np.fft.fftfreq(len(t), d=t[1] - t[0]))
        # Plot the Fourier transform
        # plt.figure()
        plt.plot(frequencies*1e-9, np.abs(Iz0Column_fft)**2 /np.max(np.abs(Iz0Column_fft)**2),color='k',label='Bunch form factor')
        # plt.xlabel('Frequency (THz)')
        # plt.ylabel('Magnitude')
        # plt.title('Fourier Transform of Iz0 Column')
        # plt.xscale('log')
        # plt.xlim([100e-3, 5])  # Focus on positive frequencies
        # plt.xticks([1e-1,500e-3, 1,2,3,4, 5])
        # plt.gca().set_xticklabels([0.1,500e-3, 1,2,3,4, 5])
        # plt.savefig(f'{DIR}/Iz0_column_{column_without_zeros}_fft.png', dpi=300, bbox_inches='tight')
    return s, Iz0Column


def parse_input(DIR):
    res = {}
    with open(f"{DIR}/input.txt") as f2:
        # print(fp.readlines())
        for line in f2.readlines():
            line = line.strip()
            # print(line)
            params = [v for v in line.split(" ") if "=" in v]
            # print(params)
            for param in params:
                tup = param.split("=")
                res[tup[0]] = tup[1]
    for k, v, in res.items():
        try:
            res[k] = float(v)
        except ValueError:
            pass
    res["Offset"], res["MeshLength"] , res['StepY'],res['StepZ']= int(res["Offset"]), int(res["MeshLength"]), float(res['StepY']), float(res['StepZ'])
    return res


def parse_geometry(DIR,geometry='geom.txt'):
    res = {'wall': [], 'dielectric1': [], 'dielectric2': []}
    material_index = 0
    material_labels = ['wall', 'dielectric1', 'dielectric2']

    with open(f"{DIR}/{geometry}") as f:
        lines = f.readlines()

    index = 0
    while index < len(lines):
        line = lines[index].strip()

        # Skip comments and empty lines
        if not line or line.startswith('%'):
            index += 1
            continue

        # Handle number of materials line
        if material_index == 0:
            material_count = int(line)
            material_index += 1
            index += 1
            continue

        # Handle number of elements line
        num_elements_line = line.split()
        num_elements = int(num_elements_line[0])
        index += 1

        # Handle segments of lines and ellipses description line
        segments_line = lines[index].strip() # Not needed
        index += 1

        material = material_labels[material_index - 1]

        # Read the coordinates
        for _ in range(num_elements):
            coords_line = lines[index].strip()
            if coords_line and not coords_line.startswith('%'):
                coords = list(map(float, coords_line.split()))
                res[material].append(coords)
            index += 1

        material_index += 1

    return res

def parse_mon(DIR,TimeType="s",mode=0,MonitorIndex=0):
    res = {}
    if mode>9:
        filename = f"{DIR}/Monitor_m{mode}_N0{MonitorIndex}.txt"
    else:
        filename = f"{DIR}/Monitor_m0{mode}_N0{MonitorIndex}.txt"

    with open(filename) as fp:
        for i in range(4):
            line = fp.readline().strip()
            params = [v for v in line.split(" ") if "=" in v]
            for param in params:
                tup = param.split("=")
                res[tup[0]] = tup[1]
    for k, v, in res.items():
        try:
            res[k] = float(v)
        except ValueError:
            pass
    # print('timetype',TimeType)    
    if TimeType=='z':
        res['k_ct'], res["k_r"], res["k_s"] , res['r0'],res['s0'], res['h_r'], res['h_s'], res['Field'] = int(res["k_ct"]), int(res["k_r"]), int(res["k_s"]), float(res['r0']), float(res['s0']), float(res['h_r']), float(res['h_s']), str(res['Field'])
    else:
        res['k_ct'], res["k_r"], res["k_z"] , res['r0'],res['z0'], res['h_r'], res['h_z'], res['Field'] = int(res["k_ct"]), int(res["k_r"]), int(res["k_z"]), float(res['r0']), float(res['z0']), float(res['h_r']), float(res['h_z']), str(res['Field'])

    r = np.arange(res['r0'], res['r0'] + res['h_r'] * res['k_r'], res['h_r'])
    if TimeType=='z':
        s = np.arange(res['s0'], res['s0'] + res['h_s'] * res['k_s'], res['h_s'])
        dim = res["k_r"] * res["k_s"] + 1
        reshape = (res["k_r"], res["k_s"])
    else:
        z = np.arange(res['z0'], res['z0'] + res['h_z'] * res['k_z'], res['h_z'])
        dim = res["k_r"] * res["k_z"] + 1
        reshape = (res["k_r"], res["k_z"])

    df = pd.read_csv(filename, skiprows=4, header=0, names=range(dim + 1), sep=" ")
    # df = df.iloc[:nrows]  # Select the first nrows rows

    df.drop(dim, axis=1, inplace=True)

    t = []
    E2D = []

    for i, row in df.iterrows():
        t.append(float(row[0]/c))
        E2D.append(np.reshape(row[1:dim], reshape))
    E2D = np.array(E2D)

    if TimeType=='z':
        return res, t, E2D, r, s
    else:
        return res, t, E2D, r, z

class MonitorData_legacy:
    def __init__(self, res, t, Field2D, r, z_or_s, field_type, orientation, time_type):
        self.res = res
        self.t = t
        self.Field2D = Field2D
        self.r = r
        self.z_or_s = z_or_s
        self.field_type = field_type
        self.orientation = orientation
        self.time_type = time_type
    
    def __str__(self):
        return f"This is a {self.time_type}-monitor of {self.field_type}_{self.orientation}"
    @property
    def Field2D(self):
        return self._Field2D
    @Field2D.setter
    def Field2D(self, Field2D):
        self._Field2D = Field2D
        self._Field2D = np.array(self._Field2D)

class MonitorData:
    def __init__(self, res, t, Field2D, r, z_or_s, field_type, orientation, time_type):
        self.res = res
        self.t = t
        self.Field2D = Field2D
        self.r = r
        self.z_or_s = z_or_s
        self.field_type = field_type
        self.orientation = orientation
        self.time_type = time_type
    @property
    def Field2D(self):
        return self._Field2D
    @Field2D.setter
    def Field2D(self, Field2D):
        self._Field2D = Field2D
        self._Field2D = np.array(self._Field2D)

    def __str__(self):
        return f"This is a {self.time_type}-monitor of {self.field_type}_{self.orientation}"
    def CutT(self, tValueMin):
        MinTindex = np.abs(self.t -(tValueMin+self.t[0])).argmin()
        self.t = self.t[MinTindex:]
        self.Field2D = self.Field2D[:,MinTindex:,:,:]
        return self.t[MinTindex:], self.Field2D[:,MinTindex:,:,:]
    def CutTTop(self,tValueMax):
        MaxTindex = np.abs(self.t -(tValueMax+self.t[0])).argmin()
        self.t = self.t[:MaxTindex]
        self.Field2D = self.Field2D[:,:MaxTindex,:,:]
        return self.t[:MaxTindex], self.Field2D[:MaxTindex,:,:]
    
    def plot_field1D(self,r_index,z_index,DIR,preamble,mode=0):
        t = self.t
        inc = t[1] - t[0]
        Field = self.Field2D[mode,:,r_index,z_index]
        fft = (np.fft.rfft(Field))
        fft_frq = (np.fft.rfftfreq(len(Field), inc)) / 1e12
        pwr = np.abs(fft)**2

        fig, ax = plt.subplots(figsize=(4,4),dpi=200)
        ax.plot((t-t[0])*1e12,Field*1e-6, "m", label=f"{self.field_type} Field")
        ax.set_title(f"{self.field_type} field mode {mode} at z={self.z_or_s[z_index]*1e2:.2f} cm")
        ax.set_xlabel("Position (m)")
        ax.set_xlabel("Time (ps)")

        if self.field_type == 'B':
            ax.set_ylabel(fr"$cB_\{self.orientation}$ (MV/m/nc)")
        else:
            ax.set_ylabel(fr"${self.field_type}_{self.orientation}$ (MV/m/nc)")
        fig.tight_layout()
        fig.savefig(f"{DIR}/{preamble}{self.field_type}_{self.orientation}-field_z-{self.z_or_s[z_index]*1e2:.1f}cm_mode{mode}.png")

        fig2, ax2 = plt.subplots(figsize=(4,4),dpi=200)

        ax2.plot(fft_frq*1e3, pwr/np.max(pwr), label="Power Spectrum")
        ax2.set_title(f"{self.field_type} power spectrum mode {mode} at z={self.z_or_s[z_index]*1e2:.2f} cm")

        ax2.set_xlabel("Frequency (GHz)")
        ax2.set_ylabel(fr"Fourier amplitude ${self.field_type}_{self.orientation}$ (a.u.)")
        ax2.set_xlim([0, fft_frq[np.argmax(pwr)]*1e3 * 4])
        fig2.tight_layout()
        fig2.savefig(f"{DIR}/{preamble}{self.field_type}_{self.orientation}-Fourier_z-{self.z_or_s[z_index]*1e2:.1f}cm_mode{mode}.png")

        # axNew = ax2.twinx()
        # angle = np.unwrap(np.angle(fft))
        # axNew.plot(fft_frq*1e3, angle, color='r', label="Phase")
        # axNew.set_ylabel("Phase (rad)")
        # axNew.legend(loc='lower right')
        # plt.close()
        return ax, fig, ax2, fig2
    def plot_power1D(self,other,z_index,DIR,preamble,mode=0):
        t = self.t
        r = self.r
        z = self.z_or_s
        r_0=self.res['h_r']*0.5
        IntegralPower = np.zeros(len(t))
        for ii in range(len(r)):   
            # B already has a factor c inside, and it is B_phi so it has a factor r[ii]+r_0 inside as well
            # Equation: S = c/2 * int ( E_r * H_phi * r dr ) from Stupakov is in Gaussian units
            # Becomes : S = 2*pi * int ( E_r * H_phi * r dr ) in SI units
            IntegralPower +=  (2*np.pi) * self.Field2D[mode,:,ii,z_index]*  other.Field2D[mode,:,ii,z_index]/mu_0/c /(r[ii]+r_0) *r[ii] * self.res['h_r'] 

        Energy = np.trapz(IntegralPower, t)
        fig, ax = plt.subplots(figsize=(4,4),dpi=200)
        ax.plot((t-t[0])*1e12,IntegralPower)
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel('Power (W/nC^2)')
        ax.set_title(f'Power mode {mode} at z={z[z_index]*1e2:.2f} cm\nTotal Energy = {Energy*1e6:.2f} µJ/nC^2')
        ax.set_xlim([0, (t[-1]-t[0])*1e12])
        # ax.set_ylim([0, 1.1*np.max(IntegralPower)])
        fig.tight_layout()
        fig.savefig(f"{DIR}/{preamble}power_z-{self.z_or_s[z_index]*1e2:.1f}cm_mode{mode}.png")
        return fig, ax, Energy, IntegralPower
    def plot_energy1D(self,other,z_index,DIR,preamble,mode=0):
        t = self.t
        r = self.r
        z = self.z_or_s
        r_0=self.res['h_r']*0.5
        IntegralPower = np.zeros(len(t))
        for ii in range(len(r)):   
            # B already has a factor c inside, and it is B_phi so it has a factor r[ii]+r_0 inside as well
            # Equation: S = c/2 * int ( E_r * H_phi * r dr ) from Stupakov is in Gaussian units
            # Becomes : S = 2*pi * int ( E_r * H_phi * r dr ) in SI units
            try:
                IntegralPower +=  (2*np.pi) * self.Field2D[mode,:,ii,z_index]*  other.Field2D[mode,:,ii,z_index]/mu_0/c /(r[ii]+r_0) *r[ii] * self.res['h_r'] 
            except:
                print(' length of r',np.shape(r))
                print(' length of Field2D',np.shape(self.Field2D))
                print(' length of other Field2D',np.shape(other.Field2D))
        EnergyList = np.zeros(len(t))
        for ii in range(len(t)):
            EnergyList[ii] = np.trapz(np.flip(IntegralPower)[:ii], -np.flip(t)[:ii])

        Energy = np.trapz(IntegralPower, t)

        fig, ax = plt.subplots(figsize=(4,4),dpi=200)
        ax.plot((np.flip(t)-t[0])*1e12,EnergyList*1e6)
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel('Energy (µJ)')
        ax.set_title(f'Energy mode {mode} at z={z[z_index]*1e2:.2f} cm\nTotal Energy = {Energy*1e6:.2f} µJ/nC^2')
        # ax.set_xlim([0, (t[-1]-t[0])*1e12])
        # ax.set_xlim([(t[-1]-t[0])*1e12,0])
        # ax.set_ylim([0, 1.1*np.max(IntegralPower)])
        fig.tight_layout()
        fig.savefig(f"{DIR}/{preamble}energy_z-{self.z_or_s[z_index]*1e2:.1f}cm_mode{mode}.png")
        return fig, ax, Energy, IntegralPower , EnergyList
    
    def plot_EnergyDist1D(self,other,z_index,DIR,preamble,mode=0):
        r_0=self.res['h_r']*0.5
        r = self.r
        t = self.t
        z = self.z_or_s
        PowerDist = np.zeros(len(r))
        for ii in range(len(r)):   
            # Instantaneous poynting vector : S = 2*pi  E_r * H_phi in SI units
            # Divide by radial coordinate is done because of how ECHO saves B. B_phi/r = H_phi
            # Then integrate over time to get the power distribution
            try:
                PowerDist[ii] =  np.trapz( (2*np.pi) * self.Field2D[mode,:,ii,z_index]*  other.Field2D[mode,:,ii,z_index]/mu_0/c /(r[ii]+r_0) , t )
            except:
                print(' length of r',np.shape(r))
                print(' length of Field2D',np.shape(self.Field2D))
                print(' length of other Field2D',np.shape(other.Field2D))

        Energy = np.trapz(PowerDist*r, r)

        fig, ax = plt.subplots(figsize=(4,4),dpi=200)
        ax.plot(r*1e3,PowerDist)
        ax.set_xlabel('r (mm)')
        ax.set_ylabel('Energy unintegrated (J/m/nC^2)')
        ax.set_title(f'Energy dist. mode {mode} at z={z[z_index]*1e2:.2f} cm\nTotal Energy = {Energy*1e6:.2f} µJ/nC^2')
        # ax.set_xlim([0, (t[-1]-t[0])*1e12])
        # ax.set_ylim([0, 1.1*np.max(IntegralPower)])
        fig.tight_layout()

        fig.savefig(f"{DIR}/{preamble}energyDist_z-{self.z_or_s[z_index]*1e2:.1f}cm_mode{mode}.png")
        plt.close()
        return r, PowerDist
    
    def plot_field_r(self,t_index,DIR,preamble,mode=0):
        r_0=self.res['h_r']*0.5
        r = self.r
        t = self.t
        z = self.z_or_s

        fig, ax = plt.subplots(figsize=(4,4),dpi=200)
        ax.plot(r*1e3,np.abs(self.Field2D[mode,t_index,:,0])*1e-6, label=f"{self.field_type} Field")
        ax.set_xlabel('r (mm)')
        if self.field_type == 'B':
            ax.set_ylabel(fr"$cB_\{self.orientation}$ (MV/m/nc)")
        else:
            ax.set_ylabel(fr"${self.field_type}_{self.orientation}$ (MV/m/nc)")
        ax.set_title(f'Field mode {mode} at t={(t[t_index]-t[0])*1e12:.2f} ps')
        fig.tight_layout()

        fig.savefig(f"{DIR}/{preamble}{self.field_type}_{self.orientation}-field_t-{(t[t_index]-t[0])*1e12:.2f}ps_mode{mode}.png")
        plt.close()
        return 
    
    def plot_IntensityDist1D(self,other,z_index,DIR,preamble,mode=0):
        r_0=self.res['h_r']*0.5
        r = self.r
        t = self.t
        z = self.z_or_s
        IntensDist = np.zeros(len(r))
        for ii in range(len(r)):   
            try:
                IntensDist[ii] =  np.mean( (self.Field2D[mode,:,ii,z_index]**2  +  other.Field2D[mode,:,ii,z_index]**2 )  )
            except:
                print(' length of r',np.shape(r))
                print(' length of Field2D',np.shape(self.Field2D))
                print(' length of other Field2D',np.shape(other.Field2D))


        fig, ax = plt.subplots(figsize=(4,4),dpi=200)
        ax.plot(r*1e3,IntensDist)
        ax.set_xlabel('r (mm)')
        ax.set_ylabel('Intensity (a.u.)')
        ax.set_title(f'Intensity distribution of mode {mode} at z = {z[z_index]*1e2:.2f} cm')
        # ax.set_xlim([0, (t[-1]-t[0])*1e12])
        # ax.set_ylim([0, 1.1*np.max(IntegralPower)])
        fig.tight_layout()
        fig.savefig(f"{DIR}/{preamble}intensity_z-{self.z_or_s[z_index]*1e2:.1f}cm_mode{mode}.png")
        plt.close()
        return r, IntensDist

def parse_wake(DIR,mode=0):
    res = {}
    if mode>9:
        filename = f"{DIR}/wakeL_{mode}.txt"
    else:
        filename = f"{DIR}/wakeL_0{mode}.txt"

    with open(filename) as fp:
        for i in range(6):
            line = fp.readline().strip()
            if 'vertical' in line:
                newline = fp.readline().strip()
                res['h_r'] = float(newline.split()[0])
                res['offset'] = float(newline.split()[1])
            elif 'rectangular' in line:
                newline = fp.readline().strip()
                res['width'] = float(newline.split()[0])
                res['sigma_bunch'] = float(newline.split()[1])


    df = pd.read_csv(filename, skiprows=6, header=0, names=range(1), sep=" ")

    
    # df.drop(dim, axis=1, inplace=True)

    s = []
    # E = []
    kLong = []

    for i, row in df.iterrows():
        # print(row)
        # if i < :
        # s.append(float(row.name[0]))
        # print(row.name)
        # s.append(float(row[0]))
        s.append(float(row.name[0]))
        # E.append(np.average(row[1:dim]))

        # Iz0.append((row[1:dim]))
        # if i == 4:
        # print(row[1:dim])
        # kLong.append([float(x) for x in row[1:dim]])
        kLong.append(row.name[1])
    kLong = np.array(kLong)
    s = np.array(s)

    # print(s)  
    # print(np.shape(kLong))
   
    return s, kLong, res

def plot_field1D(t,E_field,mode,DIR,orientation,preamble,field_type='E',ax1=None,ax2=None):
    inc = t[1] - t[0]
    # E_field[(t-t[0])*1e12<7] = 0
    # fft = np.fft.fftshift(np.fft.fft(E_field))
    # fft_frq = np.fft.fftshift(np.fft.fftfreq(len(E_field), inc)) / 1e12
    fft = (np.fft.rfft(E_field))
    fft_frq = (np.fft.rfftfreq(len(E_field), inc)) / 1e12
    pwr = np.abs(fft)**2

    if ax1:
        plt.sca(ax1)
    else:
        plt.figure(figsize=(4,4))

    # plt.figure(figsize=(6, 5))

    plt.plot((t-t[0])*1e12,E_field*1e-6, "m", label="Electric Field")
    plt.title(f"Electric field mode {mode}")
    plt.xlabel("Position (m)")
    plt.xlabel("Time (ps)")

    plt.ylabel(f"Field {field_type}{orientation} (MV/m/nc)")
    plt.tight_layout()  
    plt.savefig(f"{DIR}/{preamble}{field_type}_field_mode{mode}_{orientation}.png")

    if ax2:

        plt.sca(ax2)
    else:
        plt.figure(figsize=(4,4))



    # plt.plot(fft_frq*1e3, pwr/np.max(pwr), label="Power Spectrum")
    plt.plot(fft_frq*1e3, pwr/np.max(pwr), label="Power Spectrum")
    plt.title(f"Power Spectrum mode {mode} of Electric Field")
    # plt.xlabel("Frequency (Th_s)")
    plt.xlim([0, 1])


    plt.xlabel("Frequency (GHz)")
    plt.ylabel(f"Fourier amplitude {field_type}{orientation} (a.u.)")
    max_freq = np.max( [fft_frq[np.argmax(pwr)]*1e3 *3, 500] )   # bound should be at least 500GHz otherwise what are we talking about
    plt.xlim([0, max_freq])
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend(loc='center left')
    plt.tight_layout()  
    plt.savefig(f"{DIR}/{preamble}{field_type}_power_mode{mode}_{orientation}.png")
    # plt.show()

    # axNew = ax2.twinx()
    # angle = np.unwrap(np.angle(fft))
    # axNew.plot(fft_frq*1e3, angle, color='r', label="Phase")
    # axNew.set_ylabel("Phase (rad)")
    # axNew.legend(loc='lower right')
    # plt.close()
    return E_field,fft_frq, fft

def plot_power1D(t,r,z,E2D,B2D,p,z_index,DIR,preamble,mode=0):
    r_0=p['h_r']*0.5
    IntegralPower = np.zeros(len(t))
    for ii in range(len(r)):   
        # B already has a factor c inside, and it is B_phi so it has a factor r[ii]+r_0 inside as well
        # Equation: S = c/2 * int ( E_r * H_phi * r dr ) from Stupakov is in Gaussian units
        # Becomes : S = 2*pi * int ( E_r * H_phi * r dr ) in SI units
        IntegralPower +=  (2*np.pi) * E2D[mode,:,ii,z_index]*  B2D[mode,:,ii,z_index]/mu_0/c /(r[ii]+r_0) *r[ii] * p['h_r'] 

    Energy = np.trapz(IntegralPower, t)
    if Energy < 1e-3:   
        print('Energy', Energy*1e6,' uJ /nC^2')
    else:
        print('Energy', Energy*1e3,' mJ /nC^2')
    fig, ax = plt.subplots(figsize=(4,4),dpi=200)
    ax.plot((t-t[0])*1e12,IntegralPower)
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('Power (W/nC^2)')
    ax.set_title(f'Power in mode {mode} at z = {z[z_index]*1e2:.2f} cm')
    ax.set_xlim([0, (t[-1]-t[0])*1e12])
    # ax.set_ylim([0, 1.1*np.max(IntegralPower)])
    fig.savefig(f'{DIR}/{preamble}Power_mode{mode}_z_{z_index}.png',dpi=200,bbox_inches='tight')
    return fig, ax, Energy, IntegralPower

def plot_EnergyDist1D(t,r,z,E2D,B2D,p,z_index,DIR,preamble,mode=0):
    r_0=p['h_r']*0.5
    PowerDist = np.zeros(len(r))
    for ii in range(len(r)):   
        # Instantaneous poynting vector : S = 2*pi  E_r * H_phi in SI units
        # Divide by radial coordinate is done because of how ECHO saves B. B_phi/r = H_phi
        # Then integrate over time to get the power distribution
        try:
            PowerDist[ii] =  np.trapz( (2*np.pi) * E2D[mode,:,ii,z_index]*  B2D[mode,:,ii,z_index]/mu_0/c /(r[ii]+r_0) , t )
        except:
            print(' length of r',np.shape(r))
            print(' length of Field2D',np.shape(E2D))
            print(' length of other Field2D',np.shape(B2D))


    fig, ax = plt.subplots(figsize=(4,4),dpi=200)
    ax.plot(r*1e3,PowerDist*1e6)
    ax.set_xlabel('r (mm)')
    ax.set_ylabel('Energy unintegrated (µJ/m)')
    ax.set_title(f'Energy distribution of mode {mode}\n at z = {z[z_index]*1e2:.2f} cm')
    # ax.set_xlim([0, (t[-1]-t[0])*1e12])
    # ax.set_ylim([0, 1.1*np.max(IntegralPower)])
    fig.tight_layout()

    fig.savefig(f'{DIR}/{preamble}EnergyDist_mode{mode}.png',dpi=200)
    plt.close()
    return r, PowerDist, fig, ax

def plot_IntensityDist1D(t,r,z,E2D_r,E2D_z,p,z_index,DIR,preamble,mode=0):
    r_0=p['h_r']*0.5
    IntensDist = np.zeros(len(r))
    for ii in range(len(r)):   
        try:
            IntensDist[ii] =  np.mean( np.sqrt(E2D_r[mode,:,ii,z_index]**2  +  E2D_z[mode,:,ii,z_index]**2 )  )
        except:
            print(' length of r',np.shape(r))
            print(' length of Field2D',np.shape(E2D_r))
            print(' length of other Field2D',np.shape(E2D_z))


    fig, ax = plt.subplots(figsize=(4,4),dpi=200)
    ax.plot(r*1e3,IntensDist)
    ax.set_xlabel('r (mm)')
    ax.set_ylabel('Intensity unintegrated (a.u.)')
    ax.set_title(f'Intensity distribution of mode {mode}\n at z = {z[z_index]*1e2:.2f} cm')
    # ax.set_xlim([0, (t[-1]-t[0])*1e12])
    # ax.set_ylim([0, 1.1*np.max(IntegralPower)])
    fig.tight_layout()
    fig.savefig(f'{DIR}/{preamble}IntensityDist_mode{mode}_z_{z_index}.png',dpi=200)
    plt.close()
    return r, IntensDist
def FindNearestAstFile(DIR,ct,StepZ):
    # List all files ending in .ast
    files = [f for f in os.listdir(DIR) if f.endswith('.ast')]
    file_pattern = r"Particles_(\d+).ast"
    TimeList = []
    FileList = []
    for file in files:
        match = re.match(file_pattern, file)
        if match:
            TimeList.append(int(match.groups()[0]))
            FileList.append(file)
        else:
            if file != "particles.ast":
                print(f"File {file} does not match the pattern")
    TimeList = np.array(TimeList)*StepZ
    # Sort the timelist and the filelist at the same time
    TimeList, FileList = zip(*sorted(zip(TimeList, FileList)))

    NearestIndex = np.argmin(np.abs(TimeList - ct))

    return FileList[NearestIndex]

def plot_xzHist(ax,zcurrent,StepZ):
    AstraFile = FindNearestAstFile("Results",zcurrent,StepZ)
    df = pd.read_csv(f"Results/{AstraFile}", delim_whitespace=True, header=None)
    x_values = np.copy(df[0])
    z_values = np.copy(df[2])
    # z_values[0] = 0
    z_values[1:] = z_values[1:] + z_values[0]
    xzhist, zedges, xedges = np.histogram2d(z_values,x_values, bins=100)
    z_middles = (zedges[1:] + zedges[:-1]) / 2
    x_middles = (xedges[1:] + xedges[:-1]) / 2
    xgrid, zgrid = np.meshgrid(x_middles, z_middles)
    # plt.imshow(xzhist.T, extent=[zedges[0]*1e3, zedges[-1]*1e3,xedges[0]*1e3, xedges[-1]*1e3], cmap=DesyBlueCmap,aspect='auto')
    norm = Normalize(vmin=0.2, vmax=np.max(xzhist)*0.8)
    xzhist_with_nan = np.where(xzhist == 0, np.nan, xzhist)
    xgrid_with_nan = np.where(xzhist == 0, np.nan, xgrid)
    zgrid_with_nan = np.where(xzhist == 0, np.nan, zgrid)
    # ax.scatter(zgrid_with_nan*1e3,xgrid_with_nan*1e3,c=xzhist_with_nan.T, cmap=DesyBlueCmap, norm=norm,s=1)
    z_values_pos = z_values[x_values>0]
    x_values_pos = x_values[x_values>0]
    ax.scatter(z_values_pos*1e3,x_values_pos*1e3, s=0.5,color="white",edgecolor=None, alpha=0.4)
    # ax.pcolormesh(zedges*1e3, xedges*1e3, xzhist_with_nan.T, cmap=DesyBlueCmap, norm=norm)

    # print('Current position:',zcurrent*1e3,"mm","average of zgrid",np.mean(zgrid)*1e3, 'Astra file:',AstraFile)
    return

def FindMaxCurrent(sigma,dz,DIR="Results"):
    files = [f for f in os.listdir(DIR) if f.endswith('.ast') and f.startswith('Particles_')]
    # find alfabetically the last file
    LastFile = sorted(files)[-1]
    file_pattern = r"Particles_(\d+).ast"
    MaxCurrent = 1
    
    df = pd.read_csv(f"Results/{LastFile}", delim_whitespace=True, header=None)
    z_values = np.copy(df[2])
    # z_values[0] = 0
    z_values[1:] = z_values[1:] + z_values[0]
    # xzhist, zedges, xedges = np.histogram2d(z_values,x_values, bins=100)
    # z_middles = (zedges[1:] + zedges[:-1]) / 2
    # x_middles = (xedges[1:] + xedges[:-1]) / 2
    # xgrid, zgrid = np.meshgrid(x_middles, z_middles)

    numSigma = 10
    charge = df[7][0]*1e-9
    nBins = round(numSigma*2*sigma/dz)
    hist, bin_edges = np.histogram(z_values, bins=nBins)
    BinMiddles = (bin_edges[:-1] + bin_edges[1:]) / 2
    dz = BinMiddles[1] - BinMiddles[0]
    hist =  - hist/dz * c * charge
    if hist.max() > MaxCurrent:
        MaxCurrent = hist.max()

    # print(LastFile)
    # print(hist)
    return MaxCurrent

def EnergyLost(DIR="Results"):
    files = [f for f in os.listdir(DIR) if f.endswith('.ast') and f.startswith('Particles_')]
    # find alfabetically the last file
    FirstFile = sorted(files)[0]
    LastFile = sorted(files)[-1]
    file_pattern = r"Particles_(\d+).ast"
    MaxCurrent = 1
    df = pd.read_csv(f"Results/{FirstFile}", delim_whitespace=True, header=None)
    
    InitialMomentum = np.mean(df[5][1:])+df[5][0]

    df = pd.read_csv(f"Results/{LastFile}", delim_whitespace=True, header=None)
    
    FinalMomentum = np.mean(df[5][1:])+df[5][0]

    return InitialMomentum,FinalMomentum

def plot_zHist(ax,zcurrent,StepZ,sigma,dz,PlotFactor):
    AstraFile = FindNearestAstFile("Results",zcurrent,StepZ)
    df = pd.read_csv(f"Results/{AstraFile}", delim_whitespace=True, header=None)
    z_values = np.copy(df[2])
    # z_values[0] = 0
    z_values[1:] = z_values[1:] + z_values[0]
    # xzhist, zedges, xedges = np.histogram2d(z_values,x_values, bins=100)
    # z_middles = (zedges[1:] + zedges[:-1]) / 2
    # x_middles = (xedges[1:] + xedges[:-1]) / 2
    # xgrid, zgrid = np.meshgrid(x_middles, z_middles)

    numSigma = 10
    charge = df[7][0]*1e-9
    nBins = round(2*numSigma*sigma/dz)
    nBins = round(numSigma*2*sigma/dz)
    hist, bin_edges = np.histogram(z_values, bins=nBins)
    BinMiddles = (bin_edges[:-1] + bin_edges[1:]) / 2
    dz = BinMiddles[1] - BinMiddles[0]
    hist =  - hist/dz * c * charge
    hist = hist * PlotFactor

    ax.plot(BinMiddles*1e3, -hist,linewidth=2,  linestyle='-', color="red")

    return

def plot_field2D_zTime(t, resGeom,resInput,s,r, E2D, mode,DIR,orientation,preamble,r_indexMax=-1,s_indexMax=-1,LinePlotter=True,Step=5,MovieDuration=15):
    ii = 0
    sigma = 1.2*1e-3
    dz = sigma/10
    MaxCurrent = FindMaxCurrent(sigma,dz,DIR="Results")
    images = []
    intermediate_dir = os.path.join(DIR, "Intermediate")
    if not os.path.exists(intermediate_dir):
        os.makedirs(intermediate_dir)

    WallCoords = np.array(resGeom['wall'])

    zListWall = np.array([WallCoords[ii][j] for ii in range(len(WallCoords)) for j in [0, 2]])
    rListWall = np.array([WallCoords[ii][j] for ii in range(len(WallCoords)) for j in [1, 3]])
    rListWall = np.append(rListWall,rListWall[0])
    zListWall = np.append(zListWall,zListWall[-1])

    rListWall = rListWall*1e-2
    zListWall = zListWall*1e-2

    # WallCoords = np.array(resGeom['wall'])
    DielectricCoords = np.array(resGeom['dielectric1'])
    zListDielectric = np.array([DielectricCoords[ii][j] for ii in range(len(DielectricCoords)) for j in [0, 2]])
    rListDielectric = np.array([DielectricCoords[ii][j] for ii in range(len(DielectricCoords)) for j in [1, 3]])

    rListDielectric = rListDielectric*1e-2
    zListDielectric = zListDielectric*1e-2 

    for t_index in range(0,len(t),Step):
        plt.figure(figsize=(6,4))
        zCurrent = c*t[t_index]
        zList = zCurrent - s[s_indexMax:0:-1]
        
        # plt.imshow((E2D[t_index,0:r_indexMax,s_indexMax:0:-1])*1e-6, extent=[1e3*(zCurrent-s[s_indexMax]),1e3*zCurrent, 1e3*r[0], 1e3*r[r_indexMax]], origin='lower', aspect='auto',cmap=cmap, vmin=-np.abs(E2D).max()*0.1e-6, vmax=np.abs(E2D).max()*0.1e-6)
        NormImshow=SymLogNorm(linthresh=1e-6 * 0.10*np.abs(E2D).max(),vmin=-np.abs(E2D).max()*1e-6, vmax=np.abs(E2D).max()*1e-6)
        plt.imshow((E2D[t_index,0:r_indexMax,s_indexMax:0:-1])*1e-6, extent=[1e3*(zCurrent-s[s_indexMax]),1e3*zCurrent, 1e3*r[0], 1e3*r[r_indexMax]], origin='lower', aspect='auto',cmap=cmap, norm=NormImshow)
        # Now I plot the image mirrored over the vertical axis if orientation is "r"
        if orientation == "z":
            plt.imshow( (E2D[t_index,0:r_indexMax,s_indexMax:0:-1])*1e-6, extent=[1e3*(zCurrent-s[s_indexMax]),1e3*zCurrent, 1e3*r[0], -1e3*r[r_indexMax]], origin='lower', aspect='auto',cmap=cmap, norm=NormImshow )
        else:
            plt.imshow( - (E2D[t_index,0:r_indexMax,s_indexMax:0:-1])*1e-6, extent=[1e3*(zCurrent-s[s_indexMax]),1e3*zCurrent, 1e3*r[0], -1e3*r[r_indexMax]], origin='lower', aspect='auto',cmap=cmap, norm=NormImshow )

        # Colorbar
        # plt.colorbar(label='Electric Field (MV/m/nc)')

        if LinePlotter:
            BunchLength = resInput['BunchSigma']
            # plt.scatter((zCurrent)*1e3,100e-6 *1e3,marker='o',color='r')
            plt.plot([(zCurrent)*1e3,(zCurrent-10*BunchLength)*1e3],[resInput['Offset']*resInput['StepY']*1e3,resInput['Offset']*resInput['StepY']*1e3],linestyle='-',color='darkgoldenrod',linewidth=2,label='Bunch')
        
        # Geometry
        plt.fill(zListWall*1e3,rListWall*1e3,linestyle='-',color='white',label='Wall',fill=False,hatch='/')
        # plt.plot(np.array([zList,zList])*1e3,np.array([DielectricHeight1,DielectricHeight2])*1e3,linestyle='--',color='k',alpha=0.8)
        # plt.fill(zListDielectric*1e3,rListDielectric*1e3,linestyle='--',edgecolor='k',facecolor=None,label='Dielectric',fill=False)
        plt.fill(zListDielectric*1e3,rListDielectric*1e3,linestyle='--',edgecolor="chartreuse",label='Dielectric',alpha=0.25,facecolor="chartreuse")
        # Plot mirrored image
        plt.fill(zListWall*1e3,-rListWall*1e3,linestyle='-',color='white',fill=False,hatch='/')
        plt.fill(zListDielectric*1e3,-rListDielectric*1e3,linestyle='--',edgecolor="chartreuse",alpha=0.25,facecolor="chartreuse")
            
        plt.title(fr"$E_{orientation}$ of mode {mode} at $ct$ = {c*t[t_index]*1e3:.2f} mm")
        # plt.title(fr"$E_{orientation}$ of mode 1 at $ct$ = {c*t[t_index]*1e3:.2f} mm")
        plt.xlabel("s (mm)")
        # plt.ylim([1e3*r[0], 1e3*r[r_indexMax]])
        ax = plt.gca()
        plot_xzHist(ax,zCurrent,resInput['StepZ'])
        plot_zHist(ax,zCurrent,resInput['StepZ'],sigma,dz,0.5*rListDielectric.min()*1e3 / MaxCurrent)

        plt.ylim([- 1e3*r[r_indexMax], 1e3*r[r_indexMax]])
        plt.xlim([1e3*(zCurrent-s[s_indexMax]),1e3*zCurrent])

        

        # place xticks at 0.25, 0.5 and 0.75 of the x-axis
        xTicks = np.linspace(1e3*(zCurrent-s[s_indexMax]),1e3*zCurrent,4)
        # plt.xticks(plt.xticks()[0][:-1])
        # plt.xticks(xTicks)
        xticklabels = [f"{x:.0f}" for x in xTicks]
        plt.xticks(xTicks, xticklabels)
        plt.ylabel("r (mm)")
        plt.legend(loc='lower left')
        plt.tight_layout()
        
        plt.savefig(f"{DIR}/Intermediate/E2D_zTime_{orientation}_mode{mode}_t_{t_index}.png",dpi=200)
        
        images.append(imageio.imread(f"{DIR}/Intermediate/E2D_zTime_{orientation}_mode{mode}_t_{t_index}.png"))
        plt.close()
        ii += 1
    
    nFrames = len(images)
    Duration = MovieDuration/nFrames
    FPS = 1/Duration
    imageio.mimsave(f"{DIR}/{preamble}E2D_zTime_mode{mode}_{orientation}.gif", images, duration=Duration)
    imageio.mimsave(f"{DIR}/{preamble}E2D_zTime_mode{mode}_{orientation}.mp4", images, fps=FPS)
    # Delete all files in DIR/Intermediate/
    file_list = os.listdir(f"{DIR}/Intermediate/")
    for file_name in file_list:
        file_path = os.path.join(f"{DIR}/Intermediate/", file_name)
        os.remove(file_path)

def plot_field_Gif(t,E_field,mode,axField,axPower):
    # E_field[(t-t[0])*1e12<7] = 0
    fft = (np.fft.rfft(E_field))
    # print(t)
    inc = t[1] - t[0]
    fft_frq = ( np.fft.rfftfreq(len(E_field), inc) / 1e12 )
    pwr = np.abs(fft)**2 /np.abs(fft).max()**2

    axField.plot(t*1e12,E_field, "m", label="Electric Field")
    axField.set_ylabel("Electric Field (MV/m/nc)")
    axField.set_xlabel("Time (ps)")
    # print(fft_frq)
    # print(pwr)

    axPower.plot(fft_frq*1e3, pwr,"m" ,label="Power Spectrum")
    axPower.set_xlabel("Frequency (GHz)")
    axPower.set_ylabel("Fourier amplitude (a.u.)")
    axPower.set_xlim([0, 1000])
    return

def plot_field2D_sTime(t, resGeom,resInput,z,r, E2D, mode,r_index,z_index,DIR,orientation,preamble,r_indexMax=-1,z_indexMax=-1,LinePlotter=True,Step=20):

    ii = 0
    images = []
    intermediate_dir = os.path.join(DIR, "Intermediate")
    if not os.path.exists(intermediate_dir):
        os.makedirs(intermediate_dir)

    WallCoords = np.array(resGeom['wall'])

    zListWall = np.array([WallCoords[ii][j] for ii in range(len(WallCoords)) for j in [0, 2]])
    rListWall = np.array([WallCoords[ii][j] for ii in range(len(WallCoords)) for j in [1, 3]])

    # WallCoords = np.array(resGeom['wall'])
    DielectricCoords = np.array(resGeom['dielectric1'])
    zListDielectric = np.array([DielectricCoords[ii][j] for ii in range(len(DielectricCoords)) for j in [0, 2]])
    rListDielectric = np.array([DielectricCoords[ii][j] for ii in range(len(DielectricCoords)) for j in [1, 3]])

    for t_index in range(0,len(t),Step):
        fig, (ax2D, axField, axPower) = plt.subplots(3, 1, figsize=(4, 8))

        zCurrent = c*t[t_index]
        zList = zCurrent - z[z_indexMax:0:-1]
        zList = z[z_indexMax:0:-1]
        plt.sca(ax2D)
        # plt.imshow((E2D[t_index,0:r_indexMax,s_indexMax:0:-1])*1e-6, extent=[1e3*(zCurrent-s[s_indexMax]),1e3*zCurrent, 1e3*r[0], 1e3*r[r_indexMax]], origin='lower', aspect='auto')
        plt.imshow((E2D[t_index,0:r_indexMax,0:z_indexMax])*1e-6, extent=[1e3*z[0], 1e3*z[z_indexMax], 1e3*r[0], 1e3*r[r_indexMax]], origin='lower', aspect='auto',cmap='viridis')
        
        plt.colorbar(label='Electric Field (MV/m/nc)')

        BunchLength = resInput['BunchSigma']
        # plt.scatter((zCurrent)*1e3,100e-6 *1e3,marker='o',color='r')
        plt.plot([(zCurrent)*1e3,(zCurrent-10*BunchLength)*1e3],[resInput['Offset']*resInput['StepY']*1e3,resInput['Offset']*resInput['StepY']*1e3],linestyle='-',color='darkgoldenrod',linewidth=2,label='Bunch')
        
        if LinePlotter:
            plt.fill(zListWall*1e3,rListWall*1e3,linestyle='-',color='r',label='Wall',fill=False,hatch='/')
            # plt.plot(np.array([zList,zList])*1e3,np.array([DielectricHeight1,DielectricHeight2])*1e3,linestyle='--',color='k',alpha=0.8)
            plt.fill(zListDielectric*1e3,rListDielectric*1e3,linestyle='--',edgecolor='k',facecolor=None,label='Dielectric',fill=False)
            plt.scatter( z[z_index]*1e3,r[r_index]*1e3,marker='o',color='g',label='Monitor')
        plt.title(fr"$E_{orientation}$ of mode {mode} at $ct$ = {c*t[t_index]*1e3:.2f} mm")
        # plt.title(fr"$E_{orientation}$ of mode 1 at $ct$ = {c*t[t_index]*1e3:.2f} mm")
        plt.xlabel("s (mm)")
        plt.ylim([1e3*r[0], 1e3*r[r_indexMax]])
        # plt.xlim([1e3*(zCurrent-z[z_indexMax]),1e3*zCurrent])
        plt.xlim([1e3*z[0], 1e3*z[z_indexMax]])
        plt.ylabel("r (mm)")
        plt.legend(loc='upper right')
        if t_index>1:
            plot_field_Gif((t[0:t_index]),E2D[0:t_index,r_index,z_index]*1e-6,mode,axField,axPower)
        
        plt.savefig(f"{DIR}/Intermediate/E2D_sTime_{orientation}_mode{mode}_t_{t_index}.png",dpi=200,bbox_inches='tight')
        
        images.append(imageio.imread(f"{DIR}/Intermediate/E2D_sTime_{orientation}_mode{mode}_t_{t_index}.png"))
        plt.close()
        ii += 1

    imageio.mimsave(f"{DIR}/{preamble}E2D_sTime_mode{mode}_{orientation}.gif", images, duration=0.2)
    imageio.mimsave(f"{DIR}/{preamble}E2D_sTime_mode{mode}_{orientation}.mp4", images, fps=5)
    # Delete all files in DIR/Intermediate/
    file_list = os.listdir(f"{DIR}/Intermediate/")
    for file_name in file_list:
        file_path = os.path.join(f"{DIR}/Intermediate/", file_name)
        os.remove(file_path)

def plot_geom(resGeom,DIR,preamble, ax2D=None):

    ii = 0
    images = []

    WallCoords = np.array(resGeom['wall'])

    zListWall = np.array([WallCoords[ii][j] for ii in range(len(WallCoords)) for j in [0, 2]])*1e-2
    rListWall = np.array([WallCoords[ii][j] for ii in range(len(WallCoords)) for j in [1, 3]])*1e-2
    rListWall = np.append(rListWall,rListWall[0])
    zListWall = np.append(zListWall,zListWall[-1])

    # WallCoords = np.array(resGeom['wall'])
    DielectricCoords = np.array(resGeom['dielectric1'])
    zListDielectric = np.array([DielectricCoords[ii][j] for ii in range(len(DielectricCoords)) for j in [0, 2]])*1e-2
    rListDielectric = np.array([DielectricCoords[ii][j] for ii in range(len(DielectricCoords)) for j in [1, 3]])*1e-2

    if ax2D is None:
        fig, ax2D = plt.subplots(figsize=(8, 4),dpi=200)
    else:
        fig = ax2D.get_figure()
        plt.sca(ax2D)

    # plt.fill(zListWall*1e2,rListWall*1e3,linestyle='-',color='r',label='Wall',fill=False,hatch='/')
    plt.fill(zListWall*1e2,rListWall*1e3,linestyle='-',color='#009FDF',label='Wall',fill=False,hatch='/')
    # plt.plot(np.array([zList,zList])*1e3,np.array([DielectricHeight1,DielectricHeight2])*1e3,linestyle='--',color='k',alpha=0.8)
    # plt.fill(zListDielectric*1e2,rListDielectric*1e3,linestyle='--',edgecolor='k',facecolor=None,label='Dielectric',fill=False)
    # plt.fill(zListDielectric*1e2,rListDielectric*1e3,linestyle='--',edgecolor='k',facecolor='#f18f4d',label='__nolegend__',fill=False)
    plt.fill(zListDielectric*1e2,rListDielectric*1e3,linestyle='-',edgecolor='#f18f4d',facecolor='#f18f4d',label='Dielectric',fill=False)
    # plt.fill(zListDielectric*1e2,rListDielectric*1e3,linestyle='--',edgecolor=None,facecolor='#f18f4d',label='Dielectric',fill=True,alpha=0.2)
    # plt.fill(zListDielectric*1e2,rListDielectric*1e3,linestyle='--',edgecolor='#f18f4d',label='Dielectric',fill=False,hatch='.')
            
    plt.title(f"Geometry")
    plt.xlabel("z (cm)")
    # plt.ylim([1e3*r[0], 1e3*r[r_indexMax]])
    # plt.xlim([1e3*z[0], 1e3*z[z_indexMax]])
    plt.ylabel("r (mm)")
    plt.legend(loc='upper right')
    plt.savefig(f"{DIR}/{preamble}geom.png",bbox_inches='tight')
    return fig, ax2D

def fLin(x, a, b):
    return a * x + b

def fSqrt(x, a, b):
    return a * np.sqrt(x) + b
# def plot_field2D_sTime(t, resGeom,resInput,z,r, E2D, mode,r_index,z_index,DIR,orientation,preamble,r_indexMax=-1,z_indexMax=-1,LinePlotter=True):
# def plot_field_Gif(t,E_field,mode,axField,axPower):

def plot_specgram(t, resInput, E_field,IMG_DIR, WindowWidth,WindowOverlapFrac,preamble,frq_min=0,frq_max=600,CurveFit=None):
    inc = t[1] - t[0]
    # Sampling Rate
    fs = 1/inc * 1e-9 # GHz

    # Window Width
    nfft = WindowWidth

    # Window Overlap
    novl = nfft//WindowOverlapFrac

    # plt.figure(figsize=(13, 5))
    plt.figure(figsize=(4, 4),dpi=200)
    # plt.subplot(121)
    # plt.subplot(111)
    # print( (t[-1]-t[0])*1e12)
    # spc, frq_spc, tim_spc, _ = plt.specgram(E_field, Fs=fs, NFFT=nfft, noverlap=novl, xextent=(0, (t[-1]-t[0])*1e12), cmap="viridis")
    spc, frq_spc, tim_spc, _ = plt.specgram(E_field, Fs=fs, NFFT=nfft, noverlap=novl, xextent=(0, (t[-1]-t[0])*1e12), cmap=cmap,scale='linear',interpolation='lanczos',vmin=0e14,vmax=1e15)
    # Normalize each column of the spectrogram
    # spc = spc / np.max(spc, axis=0)
    # spc = spc / np.max(spc)
    # print(frq_spc[0], frq_spc[-1])
    

    tim_spc = np.array(tim_spc)*1e3
    # print(tim_spc)
    if CurveFit is None:
        plt.savefig(f"{IMG_DIR}/{preamble}spec.png",bbox_inches='tight',dpi=300)
        plt.ylim([200,600])

        # plt.close()
        return plt.gca(), tim_spc, frq_spc, spc
    # tim_spc = tim_spc*1e3
    # Extract frequencies present at each timepoint and the error as
    # the inverse of intensity on the spectrogram.
    tim_fit = []
    frq_fit = []
    err_fit = []

    # frq_max = 1e12

    for i, f in enumerate(frq_spc):
        for j, TimeStep in enumerate(tim_spc[-len(spc[0]):]): 
            tim_fit.append(TimeStep)
            frq_fit.append(f)
            if frq_min < f < frq_max:
                err_fit.append(1/spc[i][j]**(1/2))
            else:
                err_fit.append(1e9)
            # if spc[i][j] > 1e-2 and f > frq_max:
            #     frq_max = f
            # if spc[i][j] > 1e-2 and f < frq_min:
            #     frq_min = f

    print('Min and max frequency', frq_min, frq_max)

    # Fit linear model to the data extracted from the spectrogram
    (a, b), _ = curve_fit(fLin, tim_fit, frq_fit, sigma=err_fit, absolute_sigma=True)

    # print(a,b)
    # Generate "experimental" frequencies and phases from the regressed paramters
    frq_fit = fLin(np.array((t-t[0])*1e12), a, b)
    phs_fit = np.cumsum(2 * np.pi * frq_fit) * inc

    #plt.xlim((tim_spc[0], tim_spc[-1]))
    #plt.ylim((frq_min * 0.5, frq_max * 1.2))
    plt.title("Spectrogram")
    plt.ylabel("Frequency (GHz)")
    # plt.ylim([200e9,400e9])
    plt.ylim([0,3000])
    plt.xlabel("Time (ps)")
    # ylim = plt.axis()[2], plt.axis()[3]


    # plt.subplot(122)
    plt.plot((t-t[0])*1e12, frq_fit, "r", linewidth=2, linestyle="-", label="Linear fit")
    plt.xlim(((t[0]-t[0])*1e12, (t[-1]-t[0])*1e12))
    # plt.ylim(ylim)
    # plt.title("Reconstructed Chirp")
    # plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (ps)")
    plt.legend()

    plt.savefig(f"{IMG_DIR}/{preamble}spec.png",bbox_inches='tight',dpi=300)
    # plt.close()
    plt.clf()

    return plt.gca(), tim_spc, frq_spc, spc, a, b, frq_fit