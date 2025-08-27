import numpy as np
import os
import sys
from scipy.constants import c
from scipy.special import jv, yv, kv, iv
import cmath
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from subprocess import call
import pandas as pd
from scipy.integrate import quad

mu0 = 1.25663706212e-6
e0 = 8.854187e-12
Z0 = 377
electron_charge = 1.602e-19
m_e = 9.11e-31 # kg
m_e_MeV = 0.511 # 0.5 MeV/c^2



# vPhase = c 
# vPhase = 0.5*c 
mur = 1 # Holds for most dielectrics

def deriv1(fun, arg1):
    SmallNum = 1e-9
    return (fun(arg1+SmallNum)-fun(arg1))/SmallNum
def deriv2(fun, arg1, arg2):
    # Derivative with respect to second argument
    SmallNum = 1e-12
    return (fun(arg1, arg2 + SmallNum)-fun(arg1, arg2))/SmallNum
def deriv3(fun, arg1, arg2, arg3):
    # Derivative with respect to third argument
    SmallNum = 1e-4
    return (fun(arg1+ arg1*SmallNum, arg2, arg3 )-fun(arg1, arg2, arg3))/(SmallNum*arg1)
def deriv3_alt(fun, arg1, arg2, arg3):
    # Derivative with respect to third argument
    SmallNum = 1e-3
    return (fun(arg1+ SmallNum, arg2, arg3 )-fun(arg1, arg2, arg3))/(SmallNum)

def deriv_general(fun, arg1, **kwargs):
    # Derivative with respect to third argument
    SmallNum = 1e-2
    # if np.real(arg1) > np.imag(arg1):
    #     SmallNum = -SmallNum
    dArg = arg1*SmallNum

    return (fun(arg1+ dArg, **kwargs)-fun(arg1, **kwargs))/(dArg)
# def kr1(omega, kz, er):
#     if hasattr(kz, "__len__"):
#         N = len(kz)
#         result = np.zeros(N) + 1j*np.zeros(N)
#         for ii in range(N):
#             if hasattr(omega, "__len__"):
#                 result[ii] = cmath.sqrt(omega[ii]**2 / c**2 - kz[ii]**2)
#             else:
#                 result[ii] = cmath.sqrt(omega**2 / c**2 - kz[ii]**2)
#     else:
#         result = cmath.sqrt(omega**2 / c**2 - kz**2)
#     return result
def kr2(omega, kz, er):
    if hasattr(kz, "__len__"):
        N = len(kz)
        result = np.zeros(N) + 1j*np.zeros(N)
        for ii in range(N):
            if hasattr(omega, "__len__"):
                result[ii] = cmath.sqrt(er * omega[ii]**2 / c**2 - kz[ii]**2)
            else:
                result[ii] = cmath.sqrt(er * omega**2 / c**2 - kz[ii]**2)
    elif hasattr(omega, "__len__"):
        result = np.zeros(len(omega)) + 1j*np.zeros(len(omega))
        for ii in range(len(omega)):
            result[ii] = cmath.sqrt(er * omega[ii]**2 / c**2 - kz**2)
    else:
        result = cmath.sqrt(er * omega**2 / c**2 - kz**2)

    return result

def kr2v(omega, vPhase, er):
    result = omega/vPhase * np.sqrt(er*(vPhase/c)**2 - 1)

    return result

def kr2v_HighGamma(omega, vPhase, er):
    result = omega/c * np.sqrt(er - 1)

    return result

def kr1(omega, kz, er):
    if np.isscalar(kz):
        return cmath.sqrt(omega**2 / c**2 - kz**2)
    else:
        return np.sqrt(omega**2 / c**2 - kz**2)
    
def kr1v(omega, vPhase):
    return omega/vPhase * np.sqrt(1-(vPhase/c)**2)
    

def J0(k,r):
    return jv(0, k*r)
def J0p(k,r):
    # This holds through the recursive relation of Bessel functions
    # Derivative taken with respect to the whole argument, so no chain rule necessary
    return -jv(1, k*r)
def J1(k,r):
    return jv(1, k*r)
def J1p(k,r):
    return deriv2(jv,1, k*r)

def Jn(k,r,n):
    return jv(n, k*r)
def Jnp(k,r,n):
    return deriv2(jv,n, k*r)

def Y0(k,r):
    return yv(0, k*r)
def Y0p(k,r):
    # This holds through the recursive relation of Bessel functions
    # Derivative taken with respect to the whole argument, so no chain rule necessary
    return -yv(1, k*r)
def Y1(k,r):
    return yv(1, k*r)
def K1(k,r):
    return kv(1, np.abs(k*r))*np.sign(k*r)
def K0(k,r):
    # check if not nan
    if np.isnan(kv(0, np.abs(k*r))):
        return 1e-12
    return kv(0, np.abs(k*r))
def K0p(k,r):
    smallnum=1e-4
    try:
        Result = (kv(0, np.abs(k*r+k*r*smallnum))-kv(0, np.abs(k*r)))/(smallnum*k*r)
    except:
        return 0
    if np.isnan(Result):
        return 1e-12 * np.sign(k*r)
    return Result

def I0(k,r):
    return iv(0, np.abs(k*r))
def I0p(k,r):
    smallnum=1e-4
    return (iv(0, np.abs(k*r)+np.abs(k*r)*smallnum)-iv(0, np.abs(k*r)))/(smallnum*k*r)

def eta(omega,charge,vPhase=c):
    # try:
    return - 1j * charge *omega/(np.pi * vPhase**2)
    # except:
    #     return 0

def eta0p(omega,k1,gamma,charge,r0=0,vPhase=c):
    return eta(omega,charge,vPhase) * I0(k1,r0)/gamma**2

def F00(k,r, b):
    return jv(0,k*r) - (jv(0,k*b)/yv(0,k*b)) * yv(0,k*r)
def F00p(k,r, b):
    return -jv(1,k*r) + (jv(0,k*b)/yv(0,k*b)) * yv(1,k*r)
def F0n(k,r,b,nmode=0):
    return jv(nmode,k*r) - (jv(nmode,k*b)/yv(nmode,k*b)) * yv(nmode,k*r)
def F0np(k,r,b,nmode=0):
    return deriv2(jv,nmode,k*r) - (jv(nmode,k*b)/yv(nmode,k*b)) * deriv2(yv,nmode,k*r)
def F11(k,r,b):
    return jv(1,k*r) - (jv(1,k*b)/yv(1,k*b)) * yv(1,k*r)
def F11p(k,r,b):
    return deriv2(jv, 1,k*r) - (jv(1,k*b)/yv(1,k*b)) * deriv2(yv,1,k*r)
def G00(k,r,b):
    return jv(0,k*r) - (jv(1,k*b)/yv(1,k*b)) * yv(0,k*r)
def G00p(k,r,b):
    return -jv(1,k*r) + (jv(1,k*b)/yv(1,k*b)) * yv(1,k*r)
def G11(k,r,b):
    return jv(1,k*r) - (deriv2(jv,1,k*b)/deriv2(yv,1,k*b)) * yv(1,k*r)
def G11p(k,r,b):
    return deriv2(jv,1,k*r) - (deriv2(jv,1,k*b)/deriv2(yv,1,k*b)) * deriv2(yv,1,k*r)
def p0(k,r,b):
    return J0(k,b) * Y0(k,r) - J0(k,r) * Y0(k,b)
def p0p(k,r,b):
    return J0(k,b) * Y0p(k,r) - J0p(k,r) * Y0(k,b)
def p0p2(k,r,b):
    return J0p(k,b) * Y0(k,r) - J0(k,r) * Y0p(k,b)
def p0pp(k,r,b):
    return J0p(k,b) * Y0p(k,r) - J0p(k,r) * Y0p(k,b)




def p0(k,r,b):
    return J0(k,b) * Y0(k,r) - J0(k,r) * Y0(k,b)
def p0p(k,r,b):
    a = np.complex128(J0(k, b))
    b_ = np.complex128(Y0p(k, r))
    c = np.complex128(J0p(k, r))
    d = np.complex128(Y0(k, b))

    term1 = np.multiply(a, b_)
    term2 = np.multiply(c, d)

    return term1 - term2

def D0x(omega, kz, a, b, er):
    x = omega*b*np.sqrt(er-1)/c
    s = kr2(omega, kz, er)
    # s = omega*np.sqrt(er-1)/c

    xi = a/b
    return x*p0p(s, a, b) + x**2 * xi /(2*er) * p0(s, a, b)

def D0x_s(s, a, b, er):
    xi = a/b
    return s*b*p0p(s, a, b) + (s*b)**2 * xi /(2*er) * p0(s, a, b)

def D0x_ds(s, a, b, er):
    SmallNum = 1e-4
    return (D0x_s(s+SmallNum, a, b, er) - D0x_s(s, a, b, er))/SmallNum

def D0xp(omega, kz, a, b, er):
    # x = omega*a*np.sqrt(er-1)/c
    x = omega*b*np.sqrt(er-1)/c
    s = kr2(omega, kz, er)
    xi = a/b
    return p0p(s, a, b) + x * xi /(er) * p0(s, a, b)

def D0xp_new(omega, kz, a, b, er):
    # x = omega*a*np.sqrt(er-1)/c
    x = omega*b*np.sqrt(er-1)/c
    # s = kr2(omega, kz, er)
    s = omega*np.sqrt(er-1)/c
    xi = a/b
    # return p0p(s, a, b) + s * deriv3(p0p,s,a,b) + s*a /er * p0(s, a, b) + s**2 * a /(2*er) * deriv3(p0, s, a, b)
    return p0p(s, a, b) + x * deriv3(p0p,s,a,b)/b + x * xi/er * p0(s, a, b) + x**2 * xi /(2*er) * deriv3(p0, s, a, b)/b

def D0xp_LowGamma(omega, vPhase, a, b, er):
    # x = omega*a*np.sqrt(er-1)/c
    s = kr2v(omega, vPhase, er)
    x = b * s
    xi = a/b
    return p0p(s, a, b) + s * deriv3(p0p,s,a,b) + s*a /er * p0(s, a, b) + s**2 * a /(2*er) * deriv3(p0, s, a, b)

def Epsilon0p(omega, a, b, er,gamma,r0=0,q=1e-9,vPhase=c):
    beta = vPhase / c
    k1 = kr1v(omega, vPhase)
    s = kr2v(omega, vPhase, er)
    mu=1
    numerator = (mu* er * beta**2 -1) *k1 * K0p(k1,a)+gamma**(-2) * er * s * p0p(s,a,b) * K0(k1,a)/p0(s,a,b)
    denominator = (mu* er * beta**2 -1) * k1 * I0p(k1,a) + gamma**(-2) * er * s * p0p(s,a,b) * I0(k1,a)/p0(s,a,b)

    # print('numerator', numerator)
    # print('denominator', denominator)

    if np.isnan(numerator) or np.isnan(denominator):
        print('numerator', numerator)
        print('denominator', denominator)
        print('p0', p0(s,a,b))
        print('p0p', p0p(s,a,b))
        print('Omega', omega)
        return 0
    return -eta0p(omega,k1,gamma,q,r0,vPhase) * numerator/denominator 

def Epsilon0p_Approx(omega, a, b, er,gamma,r0=0,q=1e-9,vPhase=c):
    beta = vPhase / c
    k1 = kr1v(omega, vPhase)
    s = kr2v(omega, vPhase, er)
    mu=1
    # numerator = (mu* er  -1) *k1 * K0p(s,a)+gamma**(-2) * er * s * p0p(s,a,b) * K0(s,a)/p0(s,a,b)
    numerator = (mu* er  -1) *k1 * (-1/s*a) * p0(s,a,b) +gamma**(-2) * er * s * p0p(s,a,b) * K0(s,a)
    denominator = (mu* er  -1) * k1 * I0p(s,a) + gamma**(-2) * er * s * p0p(s,a,b) * I0(s,a)/p0(s,a,b)

    # print('numerator', numerator)
    # print('denominator', denominator)

    if np.isnan(numerator) or np.isnan(denominator):
        print('numerator', numerator)
        print('denominator', denominator)
        print('p0', p0(s,a,b))
        print('p0p', p0p(s,a,b))
        print('Omega', omega)
        return 0
    return -eta0p(omega,k1,gamma,q,r0,vPhase) * numerator/denominator 

def FourierAmplitudeLowGamma(omega, a, b, er,gamma,q=1e-9,r0=0,vPhase=c,rMeas=0):
    # if omega < 0:
    #     omega = -omega
    Eps0p = Epsilon0p(omega, a, b, er,gamma,r0=r0,q=q,vPhase=vPhase)
    k1 = kr1v(omega, vPhase)
    # print('k1', k1, 'vs', kr1(omega, omega/vPhase, er))
    # print('s', kr2v(omega, vPhase, er), 'vs', kr2(omega, omega/vPhase, er))

    CGS2SI = 1/(4*np.pi*e0)
    # print('Eps0p', Eps0p)
    # print('eta0p', eta0p(omega,k1,gamma,q,r0,vPhase))
    # print('I0', I0(k1,r0))
    # print('K0', K0(k1,r0))

    # return CGS2SI * ( Eps0p * I0(k1,r0) )
    #print('epsp', Eps0p)
    #print('k1', k1)
   # print('eta0p', eta0p(omega,k1,gamma,q,r0=r0,vPhase=vPhase) )
    #print('K0', K0(k1,rMeas) )
    return CGS2SI * ( Eps0p * I0(k1,rMeas) + eta0p(omega,k1,gamma,q,r0=r0,vPhase=vPhase) * K0(k1,rMeas) )

def FourierAmplitudeHighGamma(omega, a, b, er,gamma,q=1e-9,r0=0,vPhase=c,rMeas=0):
    # if omega < 0:
    #     omega = -omega
    k1 = kr1v(omega, vPhase)

    CGS2SI = 1/(4*np.pi*e0)
    Prefactor = eta(omega,q,vPhase=vPhase) * I0(k1,r0)
    mu=1
    s = kr2v(omega, vPhase, er)
    FirstTerm = (mu*er -1)*p0(s,a,b)/(p0p(s,a,b) + (s*a/(2*er)) * p0(s,a,b)) * I0(k1,rMeas)/(er*s*a)
    #FirstTerm = (mu*er -1)*p0(s,a,b)/(p0p(s,a,b) + (s*a/(2*(er-1e-3*1j))) * p0(s,a,b)) * I0(k1,rMeas)/(er*s*a)
    SecondTerm = K0(k1,rMeas)/gamma**2
    # SecondTerm = 0

    # print('Prefactor', Prefactor)
    # print('FirstTerm', FirstTerm)
    # print('p0', p0(s,a,b))
    # print('p0p', p0p(s,a,b))
    # print('SecondTerm', SecondTerm)
    # return (CGS2SI * Prefactor*(FirstTerm))
    return (CGS2SI * Prefactor*(FirstTerm - SecondTerm))

def FourierAmplitudeMaxGamma(omega, a, b, er,gamma,q=1e-9,r0=0,vPhase=c,rMeas=0):
    # if omega < 0:
    #     omega = -omega
    k1 = kr1v(omega, vPhase)

    CGS2SI = 1/(4*np.pi*e0)
    mu=1
    Prefactor = -1j*q*np.sqrt(mu*er-1)/(np.pi*er*a*c)
    s = kr2v(omega, vPhase, er)
    Term = p0(s,a,b)/(p0p(s,a,b) + (s*a/(2*er)) * p0(s,a,b))
    #Term = p0(s,a,b)/(p0p(s,a,b) + (s*a/(2*(er-1e-4*1j))) * p0(s,a,b))

    # print('Prefactor', Prefactor)
    # print('FirstTerm', FirstTerm)
    # print('p0', p0(s,a,b))
    # print('p0p', p0p(s,a,b))
    # print('SecondTerm', SecondTerm)
    return (CGS2SI * Prefactor*(Term))

def AmplitudeLowGamma(omega, a, b, er,gamma,q=1e-9,r0=0,tSpace=None,vPhase=c,rMeas=0):
    OmegaList=np.linspace(-omega*3,omega*3,5000)
    OmegaList=np.linspace(1e-4,omega*3,5000)
    # TotalIntegral = 0
    z=0
    t=1e-12
    t=0
    if tSpace is None:
        tSpace = np.linspace(0, 5/(omega/2/np.pi), 100)
    IntegralList = np.zeros(len(tSpace),dtype=complex)
    for ii, t in enumerate(tSpace):
        Integral = quad(lambda x: np.real(FourierAmplitudeLowGamma(x, a, b, er,gamma,q,r0,vPhase,rMeas) * np.exp(1j* (z-vPhase*t)*x/vPhase)), 0, omega*50)
        IntegralIm = quad(lambda x: np.imag(FourierAmplitudeLowGamma(x, a, b, er,gamma,q,r0,vPhase,rMeas) * np.exp(1j* (z-vPhase*t)*x/vPhase)), 0, omega*50)
        # TotalIntegral += Integral[0]
        #yList = np.zeros_like(OmegaList, dtype=complex)
       # for jj, omega in enumerate(OmegaList):
        #    yList[jj] = FourierAmplitudeLowGamma(omega, a, b, er,gamma,q=q,r0=r0,vPhase=vPhase,rMeas=rMeas) * np.exp(1j* (z-vPhase*t)*omega/vPhase)
        #print('ylist', yList)
        #Integral = np.trapz( yList, OmegaList  )
        IntegralList[ii] = ( Integral[0] + 1j*IntegralIm[0])
        #IntegralList[ii] =2*(Integral)
    print('Max argument at t = ', tSpace[np.argmax(np.abs(IntegralList))]*1e12, 'ps out of maximum', tSpace[-1]*1e12, 'ps')
   
    return tSpace,IntegralList

def AmplitudeHighGamma(omega, a, b, er,gamma,q=1e-9,r0=0,tSpace=None,vPhase=c,rMeas=0):
    OmegaList=np.linspace(-omega*3,omega*3,5000)
    OmegaList=np.linspace(0.0001,omega*3,5000)
    # TotalIntegral = 0
    z=0
    if tSpace is None:
        tSpace = np.linspace(0, 5/(omega/2/np.pi), 100)
    IntegralList = np.zeros(len(tSpace),dtype=complex)
    for ii, t in enumerate(tSpace):
        #Integral = quad(lambda x: np.real(FourierAmplitudeHighGamma(x, a, b, er,gamma,q=q,r0=r0,vPhase=vPhase,rMeas=rMeas) * np.exp(1j* (z-vPhase*t)*x/vPhase)), -omega*50, omega*50)
        #IntegralIm = quad(lambda x: np.imag(FourierAmplitudeHighGamma(x, a, b, er,gamma,q=q,r0=r0,vPhase=vPhase,rMeas=rMeas) * np.exp(1j* (z-vPhase*t)*x/vPhase)), -omega*50, omega*50)
        # TotalIntegral += Integral[0]
        yList = np.zeros_like(OmegaList, dtype=complex)
        for jj, omega in enumerate(OmegaList):
            yList[jj] = FourierAmplitudeHighGamma(omega, a, b, er,gamma,q=q,r0=r0,vPhase=vPhase,rMeas=rMeas) * np.exp(1j* (z-vPhase*t)*omega/vPhase)
        Integral = np.trapz( yList, OmegaList  )
        # IntegralList[ii] = ( Integral[0] + 1j*IntegralIm[0])
        IntegralList[ii] = 2*Integral
    print('Max argument at t = ', tSpace[np.argmax(np.abs(IntegralList))]*1e12, 'ps out of maximum', tSpace[-1]*1e12, 'ps')
    
    return tSpace,IntegralList

def AmplitudeMaxGamma(omega, a, b, er,gamma,q=1e-9,r0=0,tSpace=None,vPhase=c,rMeas=0):
    OmegaList=np.linspace(-omega*3,omega*3,100000)
    # TotalIntegral = 0
    z=0
    if tSpace is None:
        tSpace = np.linspace(0, 5/(omega/2/np.pi), 100)
    IntegralList = np.zeros(len(tSpace),dtype=complex)
    for ii, t in enumerate(tSpace):
        Integral = quad(lambda x: np.real(FourierAmplitudeMaxGamma(x, a, b, er,gamma,q=q,r0=r0,vPhase=vPhase,rMeas=rMeas) * np.exp(1j* (z-vPhase*t)*x/vPhase)), -omega*50, omega*50)
        IntegralIm = quad(lambda x: np.imag(FourierAmplitudeMaxGamma(x, a, b, er,gamma,q=q,r0=r0,vPhase=vPhase,rMeas=rMeas) * np.exp(1j* (z-vPhase*t)*x/vPhase)), -omega*50, omega*50)
        # TotalIntegral += Integral[0]
       # yList = np.zeros_like(OmegaList, dtype=complex)
        #for jj, omega in enumerate(OmegaList):
        #    yList[jj] = FourierAmplitudeMaxGamma(omega, a, b, er,gamma,q=q,r0=r0,vPhase=vPhase,rMeas=rMeas) * np.exp(1j* (z-vPhase*t)*omega/vPhase)
        #Integral = np.trapz( yList, OmegaList  )
        IntegralList[ii] = ( Integral[0] + 1j*IntegralIm[0])
        #IntegralList[ii] = Integral
    print('Max argument at t = ', tSpace[np.argmax(np.abs(IntegralList))]*1e12, 'ps out of maximum', tSpace[-1]*1e12, 'ps')
    
    return tSpace,IntegralList

def Amplitude(omega, kz, a, b, er,q=1e-9):
    # x = omega*a*np.sqrt(er-1)/c
    x = omega*b*np.sqrt(er-1)/c
    s = kr2(omega, kz, er)
    xi = a/b
    diff = 1e-4
    diffX = diff*b*np.sqrt(er-1)/c
    # deriv = (D0x(omega+diff, (omega+diff)/c, a, b, er) - D0x(omega, kz, a, b, er))/diffX
    # deriv = D0xp(omega, kz, a, b, er)
    deriv = D0xp_new(omega, kz, a, b, er)
    # print('deriv', deriv)
    # prefactor = 4*q/(a*b*er)
    prefactor = 4*q/(a*er)
    return prefactor*x*p0(s, a, b)/deriv

def AmplitudeSI(omega, kz, a, b, er,q=1e-9):
    # x = omega*a*np.sqrt(er-1)/c
    x = omega*b*np.sqrt(er-1)/c
    # s = kr2(omega, kz, er)
    s = omega*np.sqrt(er-1)/c

    xi = a/b
    diff = 1e-4
    diffX = diff*b*np.sqrt(er-1)/c
    # deriv = (D0x(omega+diff, (omega+diff)/c, a, b, er) - D0x(omega, kz, a, b, er))/diffX
    # deriv = D0xp(omega, kz, a, b, er)
    deriv = D0xp_new(omega, kz, a, b, er)
    # print('deriv', deriv)
    prefactor = 4*q/(a*b*er)
    # CGS2SI = Z0*c/(4*np.pi)
    CGS2SI = 1/(4*np.pi*e0)
    return CGS2SI  *prefactor*x*p0(s, a, b)/deriv
    # return b * Z0*c  *prefactor*x*p0(s, a, b)/deriv

    NormalizationCGS2SI=4.0/a*electron_charge/(4*np.pi*er*e0)
    return NormalizationCGS2SI* prefactor*x*p0(s, a, b)/deriv

def AmplitudeSI_LowGamma(omega, vPhase, a, b, er,q=1e-9):
    # x = omega*a*np.sqrt(er-1)/c
    # s = kr2(omega, kz, er)
    s = kr2v(omega, vPhase, er)
    x = s * b
    xi = a/b
    # deriv = (D0x(omega+diff, (omega+diff)/c, a, b, er) - D0x(omega, kz, a, b, er))/diffX
    # deriv = D0xp(omega, kz, a, b, er)
    deriv = D0x_ds(s, a, b, er)/b
    # print('deriv', deriv)
    prefactor = 4*q/(a*b*er)
    # CGS2SI = Z0*c/(4*np.pi)
    CGS2SI = 1/(4*np.pi*e0)
    return CGS2SI  *prefactor*x*p0(s, a, b)/deriv
    # return b * Z0*c  *prefactor*x*p0(s, a, b)/deriv

    NormalizationCGS2SI=4.0/a*electron_charge/(4*np.pi*er*e0)
    return NormalizationCGS2SI* prefactor*x*p0(s, a, b)/deriv

def AmplitudeSI_Klaus(a,q=1e-9):
    # only valid for high frequencies
    # x = omega*a*np.sqrt(er-1)/c
    # s = kr2(omega, kz, er)
    LossFactor = -Z0 * c/(2*np.pi * a**2)
    return LossFactor
    # return b * Z0*c  *prefactor*x*p0(s, a, b)/deriv

    # NormalizationCGS2SI=4.0/a*electron_charge/(4*np.pi*er*e0)
    # return NormalizationCGS2SI* prefactor*x*p0(s, a, b)/deriv

def g_old(k2,a,b):
    TopFrac = J1(k2,a) * Y0(k2,b) - J0(k2,b) * Y1(k2,a)

    BottomFrac = J0(k2,a) * Y0(k2,b) - J0(k2,b) * Y0(k2,a)
    # print('result', TopFrac/BottomFrac)
    return 1/k2 * TopFrac/BottomFrac

def g(k2,a,b,nmode=0):
    # print the arguments
    # print('k2', k2)
    # print('a', a)
    # print('b', b)
    # print('nmode', nmode)
    # TopFrac = -F0np(k2,a,b,nmode) * yv(nmode,k2*b)
    # TopFrac = deriv2(jv,nmode,k2*a)*yv(nmode,k2*b) - jv(nmode,k2*b) * deriv2(yv,nmode,k2*a)
    TopFrac = deriv2(jv,nmode,k2*a) - jv(nmode,k2*b)/yv(nmode,k2*b) * deriv2(yv,nmode,k2*a)
    # BottomFrac = F0n(k2,a,b,nmode) * yv(nmode,k2*b)
    BottomFrac = jv(nmode,k2*a)  - jv(nmode,k2*b)/yv(nmode,k2*b) * yv(nmode,k2*a) 
    # print('result', TopFrac/BottomFrac)
    return -1/k2 * TopFrac/BottomFrac

def h(k1,a,b,nmode=0):
    TopFrac = Jnp(k1,a,nmode)
    BottomFrac = Jn(k1,a,nmode)
    # print('result', TopFrac/BottomFrac)
    if k1 != 0:
        return 1/k1 * TopFrac/BottomFrac
    else:
        print('k1 is zero in h')
        return a**3/8 * k1**2 /2

def vG(omega, kz, a, b,er):
    k2 = kr2(omega, kz, er)
    dgdk2 = deriv3_alt(g, k2,a,b)
    # print('dgdk2', dgdk2)
    TopFrac = c*( (er/k2)*dgdk2 + a**3 /8)
    BottomFrac = (er* er/k2 * dgdk2) - a**3/8
    return np.real(TopFrac/BottomFrac)

def new_vGLowGamma(omega,kz,a,b,er,nmode=0):
    # this one holds for higher modes too but doesn't work as well for nmode=0 unfortunately
    k1 = kr1(omega, kz, er)
    k2 = kr2(omega, kz, er)
    # dgdk2 = deriv3(g, k2,a,b)
    # dgdk2 = deriv3_alt(g, k2,a,b)
    # dhdk1 = deriv3_alt(h, k1,a,b)
    dgdk2 = deriv_general(g, k2,a=a,b=b,nmode=nmode)
    dhdk1 = deriv_general(h, k1,a=a,b=b,nmode=nmode)
    
    if np.abs(k1) < 5e2:
        print('k1 is zero')
        if nmode==0:
            return vGLowGamma(omega,kz,a,b,er)
        # else:
        #     return c

    # TopFrac = - kz* (  - er/k2 * dgdk2 - 1/k1 * ( - a/k1 - (a * J1(k1,a)**2 /(k1 * J0(k1,a)**2)) + (2*J1(k1,a) /(k1**2 * J0(k1,a))  )   ))
    # BottomFrac = omega/c**2 *(  er**2/k2 * dgdk2 + 1/k1 *(  -a / k1   - (a * J1(k1,a)**2 /(k1 * J0(k1,a)**2)) + (2*J1(k1,a) /(k1**2 * J0(k1,a))  ))  )
    
    TopFrac = - kz* (  - er/k2 * dgdk2 - 1/k1 * dhdk1)
    BottomFrac = omega/c**2 *(  er**2/k2 * dgdk2 + 1/k1 * dhdk1  )
    
    # print('Topfrac', TopFrac)
    # print('bottomfrac', BottomFrac)
    return np.real(TopFrac/BottomFrac) if np.real(TopFrac/BottomFrac) < c else c



def vGLowGamma(omega,kz,a,b,er,nmode=0):
    k1 = kr1(omega, kz, er)
    k2 = kr2(omega, kz, er)
    # dgdk2 = deriv3(g, k2,a,b)
    dgdk2 = deriv3_alt(g_old, k2,a,b)
    # dhdk1 = deriv3_alt(h, k1,a,b)
    
    if np.abs(k1) < 1e-10:
        print('k1 is zero')
        return c

    TopFrac = - kz* (  - er/k2 * dgdk2 - 1/k1 * ( - a/k1 - (a * J1(k1,a)**2 /(k1 * J0(k1,a)**2)) + (2*J1(k1,a) /(k1**2 * J0(k1,a))  )   ))
    BottomFrac = omega/c**2 *(  er**2/k2 * dgdk2 + 1/k1 *(  -a / k1   - (a * J1(k1,a)**2 /(k1 * J0(k1,a)**2)) + (2*J1(k1,a) /(k1**2 * J0(k1,a))  ))  )
    
    # TopFrac = - kz* (  - er/k2 * dgdk2 - 1/k1 * dhdk1)
    # BottomFrac = omega/c**2 *(  er**2/k2 * dgdk2 + 1/k1 * dhdk1  )
    
    # print('Topfrac', TopFrac)
    # print('bottomfrac', BottomFrac)
    return np.real(TopFrac/BottomFrac)

def vGKlaus(a,b,er):
    return (1-4*(b-a)/a * (er-1)/er)*c

def vGLowGamma2(omega,kz,a,b,er):
    k1 = kr1(omega, kz, er)
    k2 = kr2(omega, kz, er)
    # dgdk2 = deriv3(g, k2,a,b)
    dgdk2 = deriv3_alt(g, k2,a,b)
    if k1 == 0:
        print('k1 is zero')
        return c

    TopFrac = - kz* (  - er/k2 * dgdk2 - 1/k1 * ( - a/k1 - (a * J1(k1,a)**2 /(k1 * J0(k1,a)**2)) + (2*J1(k1,a) /(k1**2 * J0(k1,a))  )   ))
    BottomFrac = omega/c**2 *(  er**2/k2 * dgdk2 + 1/k1 *(  -a / k1   - (a * J1(k1,a)**2 /(k1 * J0(k1,a)**2)) + (2*J1(k1,a) /(k1**2 * J0(k1,a))  ))  )
    # print('Topfrac', TopFrac)
    # print('bottomfrac', BottomFrac)
    return np.real(TopFrac/BottomFrac)

def vGLowGamma_vPhase(omega,vPhase,a,b,er):
    k1 = kr1v(omega, vPhase)
    k2 = kr2v(omega, vPhase, er)
    kz = omega/vPhase
    # dgdk2 = deriv3(g, k2,a,b)
    dgdk2 = deriv3_alt(g, k2,a,b)
    if k1 == 0:
        print('k1 is zero')
        return c

    TopFrac = - kz* (  - er/k2 * dgdk2 - 1/k1 * ( - a/k1 - (a * J1(k1,a)**2 /(k1 * J0(k1,a)**2)) + (2*J1(k1,a) /(k1**2 * J0(k1,a))  )   ))
    BottomFrac = omega/c**2 *(  er**2/k2 * dgdk2 + 1/k1 *(  -a / k1   - (a * J1(k1,a)**2 /(k1 * J0(k1,a)**2)) + (2*J1(k1,a) /(k1**2 * J0(k1,a))  ))  )
    # print('Topfrac', TopFrac)
    # print('bottomfrac', BottomFrac)
    return np.real(TopFrac/BottomFrac)

def ExpDistr(zz, Q, sigmaz):
    sigmat = sigmaz/c
    Bunch=Q/np.sqrt(2.*np.pi*sigmat**2)*np.exp(-(zz-sigmaz*4.0)**2/(2.*(c*sigmat)**2))
    plt.plot(zz*1e6, Bunch)
    plt.xlabel('z (µm)')
    plt.grid()
    plt.savefig('figs/Bunch.png')
    plt.close()
    return Bunch

def FindDelta(DeltaList, Inner_Radius, Omega, vPhase, er):
    D0x_store = D0x(Omega, Omega/vPhase, Inner_Radius, DeltaList+Inner_Radius, er)
    zero_crossings = np.where(np.diff(np.sign(D0x_store)))[0]
    Delta = DeltaList[zero_crossings[0]]
    return Delta

def FindRadius(Delta, RadiusList, Omega, vPhase, er):
    D0x_store = D0x(Omega, Omega/vPhase, RadiusList, Delta+RadiusList, er)
    zero_crossings = np.where(np.diff(np.sign(D0x_store)))[0]
    return RadiusList[zero_crossings[0]]

def FindWaveguideParam(OmegaList, vPhase, Inner_Radius, Layer_Thickness, er, printer=False):
    D0x_store = D0x(OmegaList, OmegaList/vPhase, Inner_Radius, Layer_Thickness+Inner_Radius, er)
    zero_crossings = np.where(np.diff(np.sign(D0x_store)))[0]
    Omega_Frequency = OmegaList[zero_crossings[0]]
    Resonance_Frequency = Omega_Frequency/(2*np.pi)
    # Resonance_Frequency = freq_0 * c/vPhase 

    D0x_store = D0x(OmegaList, OmegaList/c, Inner_Radius, Layer_Thickness+Inner_Radius, er)
    zero_crossings = np.where(np.diff(np.sign(D0x_store)))[0]
    Omega_Frequency0 = OmegaList[zero_crossings[0]]
    freq_0 = Omega_Frequency0/(2*np.pi)
    k0 = Omega_Frequency0/c

    if printer:
        print('k0', k0, '1/m')
        print('freq 0', freq_0*1e-12,'THz')
        print('Resonance frequency', Resonance_Frequency*1e-12,'THz +/-', (OmegaList[zero_crossings[0]+1]-OmegaList[zero_crossings[0]-1])/(2*np.pi)*1e-12,'THz')
        print('Resonance frequency', Resonance_Frequency*1e-9,'GHz +/-', (OmegaList[zero_crossings[0]+1]-OmegaList[zero_crossings[0]-1])/(2*np.pi)*1e-9,'GHz')
    # vGroup = vGLowGamma(Resonance_Frequency*2*np.pi, Resonance_Frequency*2*np.pi/vPhase, Inner_Radius, Layer_Thickness+Inner_Radius, er)
    # vGroup = vGLowGamma_vPhase(Resonance_Frequency*2*np.pi, vPhase, Inner_Radius, Layer_Thickness+Inner_Radius, er)
    vGroup = vGLowGamma(Resonance_Frequency*2*np.pi, Resonance_Frequency*2*np.pi/vPhase, Inner_Radius, Layer_Thickness+Inner_Radius, er,nmode=0)
    # print('vGroup in function', vGroup/c, 'vs vGroup_vphase', vGroup_vphase/c)
    return Resonance_Frequency, k0, freq_0, vGroup

def FindWaveguideModes(OmegaList, vPhase, Inner_Radius, Layer_Thickness, er):
    D0x_store = D0x(OmegaList, OmegaList/vPhase, Inner_Radius, Layer_Thickness+Inner_Radius, er)
    zero_crossings = np.where(np.diff(np.sign(D0x_store)))[0]

    ResonanceOmegas = np.zeros(len(zero_crossings))
    ModeAmplitudes = np.zeros(len(zero_crossings))
    vGroupList = np.zeros(len(zero_crossings))
    for ii, zero_crossing in enumerate(zero_crossings):
        ResonanceOmegas[ii] = OmegaList[zero_crossing]
        ModeAmplitudes[ii] = AmplitudeSI(ResonanceOmegas[ii], ResonanceOmegas[ii]/vPhase, Inner_Radius, Layer_Thickness+Inner_Radius, er,q=1)
        vGroupList[ii] = vGLowGamma(ResonanceOmegas[ii], ResonanceOmegas[ii]/vPhase, Inner_Radius, Layer_Thickness+Inner_Radius, er,nmode=ii)

    ResonanceFrequencies = ResonanceOmegas/(2*np.pi)
    return ResonanceFrequencies, ModeAmplitudes, vGroupList


def FindWaveguideParam_ECHO1D(gamma,vPhase, Inner_Radius, Layer_Thickness, er,Nmode, RunPlease=False, PlotBool=False):

    filename = f'ECHO_imps/Z_a{Inner_Radius*1e3:.2f}mm_delta{Layer_Thickness*1e6:.2f}um_eps{er}_g{gamma:.2f}_Nmode{Nmode}.dat'
    # check if file exists or bool is turned on
    if not os.path.isfile(filename) or RunPlease:
        call(["python3", "Wake4astra_ECHO1D.py", str(Inner_Radius), str(Layer_Thickness), str(er), str(4e-2), str(1.2e-3), str(gamma), str(1e-9), str(Nmode), str(PlotBool)])

    # Readout the file with pandas
    data = pd.read_csv(filename, delimiter='\s', header=0)
    freq_GHz = data[data.columns[0]]
    ReZ = data[data.columns[1]]
    ImZ = data[data.columns[2]]
    absZ = data[data.columns[3]]

    # Find the resonance frequency
    # zero_crossings = np.where(np.diff(np.sign(ImZ)))[0]
    peak_indices = []
    well_indices = []
    for i in range(1, len(ImZ)-1  ):
        if ImZ[i] > 0 and ImZ[i-1] < 0:
            well_indices.append(i)
        elif ImZ[i] < 0 and ImZ[i-1] > 0:
            peak_indices.append(i)
    # Resonance_Frequency = freq_GHz[zero_crossings[0]]*1e9
    # Resonance_Frequency = freq_GHz[np.argmax(ReZ)]*1e9
    Resonance_Frequency = freq_GHz[peak_indices[0]]*1e9
    k0 = 2*np.pi*Resonance_Frequency/c

    vGroup = vGLowGamma(Resonance_Frequency*2*np.pi, Resonance_Frequency*2*np.pi/vPhase, Inner_Radius, Layer_Thickness+Inner_Radius, er,nmode=0)

    # print('vGroup in function', vGroup/c)

    return Resonance_Frequency, k0, Resonance_Frequency, vGroup

def Mom2Ek(Momentum, Mass=m_e_MeV):
    return np.sqrt(Momentum**2 + Mass**2) - Mass
def E2Mom(Energy, Mass=m_e_MeV):
    return np.sqrt(Energy**2 - Mass**2)
def RunWake4AstraGaussian_Legacy(Inner_Radius, Layer_Thickness,TubeL, er, zmax, sigmaz, charge, Nmode, RunPlease=True,PlotBool=False):
    # Run wake4astra.py
    """a = float(sys.argv[1])
    delta = float(sys.argv[2])
    b = a + delta
    epsilon = float(sys.argv[3])
    zmax = float(sys.argv[4])
    sigmaz = float(sys.argv[5])
    Q = float(sys.argv[6])
    Nmode = int(sys.argv[7])"""

    filename = f'ECHO_wakes/w_a{Inner_Radius*1e3:.2f}mm_delta{Layer_Thickness*1e6:.2f}um_eps{er}_zmax{zmax*1e2:.2f}cm_sigmaz{sigmaz*1e3:.2f}mm_Q{charge*1e12:.2f}pC_Nmode{Nmode}.dat'
    # check if file exists or bool is turned on
    if not os.path.isfile(filename) or RunPlease:
        call(["python3", "wake4astra_param.py", str(Inner_Radius), str(Layer_Thickness), str(er), str(zmax), str(sigmaz), str(charge), str(Nmode), str(PlotBool)])

    # Readout the file with pandas
    data = pd.read_csv(filename, delimiter='\t', header=0)
    zz = data[data.columns[0]]
    Bunch = data[data.columns[1]]
    WakeGreens = data[data.columns[2]]
    WakePot = data[data.columns[3]]*TubeL



    return zz, Bunch, WakeGreens, WakePot

def RunWake4AstraGaussian(Inner_Radius, Layer_Thickness,TubeL, er, zmax, sigmaz,gamma, charge, Nmode, RunPlease=True,PlotBool=False):
    # Run wake4astra.py
    """a = float(sys.argv[1])
    delta = float(sys.argv[2])
    b = a + delta
    epsilon = float(sys.argv[3])
    zmax = float(sys.argv[4])
    sigmaz = float(sys.argv[5])
    Q = float(sys.argv[6])
    Nmode = int(sys.argv[7])"""

    filename = f'ECHO_wakes/wake_a{Inner_Radius*1e3:.2f}mm_delta{Layer_Thickness*1e6:.2f}um_eps{er}_zmax{zmax*1e2:.2f}cm_sigmaz{sigmaz*1e3:.2f}mm_Q{charge*1e12:.2f}pC_gamma{gamma:.2f}_Nmode{Nmode}.dat'
    # check if file exists or bool is turned on
    if not os.path.isfile(filename) or RunPlease:
        call(["python3", "Wake4astra_ECHO1D.py", str(Inner_Radius), str(Layer_Thickness), str(er), str(zmax), str(sigmaz), str(gamma), str(charge), str(Nmode), str(PlotBool)])

    # Readout the file with pandas
    data = pd.read_csv(filename, delimiter='\t', header=0)
    zz = data[data.columns[0]]
    Bunch = data[data.columns[1]]
    WakeGreens = data[data.columns[2]]
    WakePot = data[data.columns[3]]*TubeL


    vPhase = (1 - 1/gamma**2)**(1/2) * c

    filename = f'ECHO_imps/Z_a{Inner_Radius*1e3:.2f}mm_delta{Layer_Thickness*1e6:.2f}um_eps{er}_g{gamma:.2f}_Nmode{Nmode}.dat'
    # # check if file exists or bool is turned on
    # if not os.path.isfile(filename) or RunPlease:
    #     call(["python3", "Wake4astra_ECHO1D.py", str(Inner_Radius), str(Layer_Thickness), str(er), str(4e-2), str(1.2e-3), str(gamma), str(1e-9), str(Nmode), str(PlotBool)])

    # Readout the file with pandas
    data = pd.read_csv(filename, delimiter='\t', header=0)
    freq_GHz = data[data.columns[0]]
    ReZ = data[data.columns[1]]
    ImZ = data[data.columns[2]]
    absZ = data[data.columns[3]]

    # Find the resonance frequency
    # zero_crossings = np.where(np.diff(np.sign(ImZ)))[0]
    peak_indices = []
    well_indices = []
    for i in range(1, len(ImZ)-1  ):
        if ImZ[i] > 0 and ImZ[i-1] < 0:
            well_indices.append(i)
        elif ImZ[i] < 0 and ImZ[i-1] > 0:
            peak_indices.append(i)
    # Resonance_Frequency = freq_GHz[zero_crossings[0]]*1e9
    # Resonance_Frequency = freq_GHz[np.argmax(ReZ)]*1e9
    Resonance_Frequency = freq_GHz[peak_indices[0]]*1e9
    k0 = 2*np.pi*Resonance_Frequency/c

    vGroup = vGLowGamma(Resonance_Frequency*2*np.pi, Resonance_Frequency*2*np.pi/vPhase, Inner_Radius, Layer_Thickness+Inner_Radius, er,nmode=0)

    print('vGroup in function', vGroup/c)

    return Resonance_Frequency, k0, Resonance_Frequency, vGroup, zz, Bunch, WakeGreens, WakePot

from scipy.optimize import root_scalar

def yr(u):

    def equation(y):
        if y == 0:
            return np.inf  # Avoid division by zero
        numerator = iv(0, y * u)
        denominator = iv(1, y * u)
        return u * (numerator / denominator) - 2 * y
    # Use root-finding to solve the equation
    result = root_scalar(equation, bracket=[1e-6, 10], method='brentq')

    if result.converged:
        return result.root
    else:
        raise ValueError("Failed to converge to a solution.")

def BigK(u):
    # eq B11 in "Using pipe with corrugated walls for a subterahertz free electron laser", Stupakov
    y = yr(u) 
    numerator = 2*y * kv(1,u*y) + u*kv(0,u*y)
    denominator = -u * deriv2(iv,0,u*y) * u  + 2*y * deriv2(iv,1,u*y)  *u + 2* iv(1,u*y)
    # return u**2 * y * numerator/denominator  /(2*y**4)
    return u**2 * y * numerator/denominator

def E1(Kappa, Q, F):
    return Kappa * Q * F

def E2(omega,kz,a,b,er,Kappa, Q, F):
    k2 = kr2(omega,kz,er)
    return E1(Kappa, Q, F) / F00(k2,a,b)
def Ip1(omega,kz,a,b,er):
    k2 = kr2(omega,kz,er)
    IntRange = np.linspace(k2*a, k2*b, 5000)
    Integral = 0
    ds = IntRange[1] - IntRange[0]
    for ii, sPos in enumerate(IntRange):
        IntVal = ( (J0(k2,b)/Y0(k2,b))*yv(1,sPos) - jv(1,sPos)   )**2 * sPos
        Integral += IntVal*ds
    return Integral

def Ip2(omega,kz,a,b,er):
    k2 = kr2(omega,kz,er)
    IntRange = np.linspace(k2*a, k2*b, 5000)
    Integral = 0
    ds = IntRange[1] - IntRange[0]
    for ii, sPos in enumerate(IntRange):
        IntVal = ( (J0(k2,b)/Y0(k2,b))*yv(0,sPos) - jv(0,sPos)   )**2 * sPos
        Integral += IntVal*ds
    return Integral

def KlausPower(kz, Q, F):
    return (Q*F)**2 * Z0 * kz**2 * c**2 /(16*np.pi)

def KlausEnergy(a,L, Q, F):
    return (Q*F)**2 * Z0 * L *c/(2*np.pi*a**2)
def ForwardPower(omega,kz,a,b,er,Kappa, Q, F):
    E1Factor = E1(Kappa, Q, F)
    E2Factor = E2(omega,kz,a,b,er,Kappa, Q, F)
    k2 = kr2(omega,kz,er)
    IpInt = Ip1(omega,kz,a,b,er)
    FirstTerm = np.pi*E1Factor**2 * a**4 * e0 * kz * omega/16
    SecondTerm = np.pi*E2Factor**2 * e0 * er * kz * omega/k2**4 * IpInt
    return FirstTerm + SecondTerm

def dPloss_dz_Cond(omega,kz,a,b,er,Kappa, Q, F, sigma0):
    E2Factor = E2(omega,kz,a,b,er,Kappa, Q, F)
    k2 = kr2(omega,kz,er)
    Prefactor = - np.pi*E2Factor**2 * b * e0**2 * er**2 * omega**2 / (np.sqrt(2)*k2**2 * sigma0 * np.sqrt(1/(mu0*omega*sigma0)))
    TermSq = ( jv(0, b*k2) * yv(1, b*k2)/yv(0, b*k2) - jv(1, b*k2) )**2
    return Prefactor * TermSq


def alphaCond(omega,kz,a,b,er,Kappa, Q, F, sigma0):
    Pprime = dPloss_dz_Cond(omega,kz,a,b,er,Kappa, Q, F, sigma0)
    Power = ForwardPower(omega,kz,a,b,er,Kappa, Q, F)
    return -Pprime/Power/2


def ForwardPower_Cond(omega,kz,a,b,er,Kappa, Q, F, sigma0,s):
    # s is the coordinate behind the bunch
    Power = ForwardPower(omega,kz,a,b,er,Kappa, Q, F)
    alpha = alphaCond(omega,kz,a,b,er,Kappa, Q, F, sigma0)
    return Power*np.exp(-alpha*s) 

def dPloss_dz_Diel(omega,kz,a,b,er,Kappa, Q, F, erprime):
    E2Factor = E2(omega,kz,a,b,er,Kappa, Q, F)
    k2 = kr2(omega,kz,er)
    IpInt = Ip1(omega,kz,a,b,er)
    Ip2Int = Ip2(omega,kz,a,b,er)
    Prefactor = -np.pi * E2Factor**2 * e0 * erprime * omega/k2**4
    Term =  IpInt * kz**2 + Ip2Int * k2**2
    return Prefactor * Term
def ForwardPower_Diel(omega,kz,a,b,er,Kappa, Q, F, erprime,s):
    # s is the coordinate behind the bunch
    Power = ForwardPower(omega,kz,a,b,er,Kappa, Q, F)
    alpha = alphaDiel(omega,kz,a,b,er,Kappa, Q, F, erprime)
    return Power*np.exp(-alpha*s)

def alphaDiel(omega,kz,a,b,er,Kappa, Q, F, erprime):
    Pprime = dPloss_dz_Diel(omega,kz,a,b,er,Kappa, Q, F, erprime)
    Power = ForwardPower(omega,kz,a,b,er,Kappa, Q, F)
    return -Pprime/Power/2

def alphaDiel(omega,kz,a,b,er,Kappa, Q, F, erprime):
    Pprime = dPloss_dz_Diel(omega,kz,a,b,er,Kappa, Q, F, erprime)
    Power = ForwardPower(omega,kz,a,b,er,Kappa, Q, F)
    return -Pprime/Power/2

def ForwardPower_Cond_Diel(omega,kz,a,b,er,Kappa, Q, F, sigma0, erprime,s):
    Power = ForwardPower(omega,kz,a,b,er,Kappa, Q, F)
    alpha = alphaCond(omega,kz,a,b,er,Kappa, Q, F, sigma0) + alphaDiel(omega,kz,a,b,er,Kappa, Q, F, erprime)
    return Power*np.exp(-alpha*s)


def EnergyStaticBunch(Inner_Radius, Layer_Thickness, TubeL, er, charge,sigmaz, zz, Bunch,WakeGreens, WakePot, Resonance_Frequency, vGroup,gamma,plotter=False, printer=False):
    U_parameter = Inner_Radius*Resonance_Frequency*2*np.pi/(gamma*c)
    LossFactorWake = np.max(WakePot)/2 
    LossFactorWake = LossFactorWake/charge/TubeL  # see other function, where it gets multiplied by tubeL
    LossFactorCorrugated = Z0*c/(4*np.pi)  * 2/(Inner_Radius**2) * BigK(U_parameter) # V/C/m
    LossFactor0 = Z0*c/(4*np.pi)  * 2/(Inner_Radius**2)


    Ez0 = - LossFactor0*charge*TubeL
    zCentral = zz[np.argmax(Bunch)]
    WakeLength = TubeL*(1 - vGroup/c)
    zzWake = zz[zz<zCentral+WakeLength]
    zzWake = zzWake[zzWake>zCentral]
    WakeTheory = Ez0 * np.cos(Resonance_Frequency*2*np.pi*(zzWake-zCentral)/c)
    LossFactorGreens = np.max(WakeGreens)/2

    if plotter:
        plt.subplots(nrows=3, sharex='all')
        plt.subplot(311)
        plt.plot (zz*1e3,WakeGreens,'-')
        plt.ylabel(r'$w(r,\zeta)$ (V/m/C)', fontsize=18)
        plt.xlabel(r'$\zeta$ (m)', fontsize=18)
        # plt.figure()
        plt.subplot(312)
        plt.plot (zz*1e3,Bunch,'-')
        plt.xlabel(r'$z$ (m)', fontsize=18)
        plt.ylabel(r'$\Lambda(z)$ (A)', fontsize=18)
        plt.xlim(0,max(zz)*1e3)
        plt.subplot(313)
        plt.plot (zz*1e3,-WakePot,'-',label='WakePot')
        plt.plot (zzWake*1e3,WakeTheory,'r--',label='WakeTheory')
        plt.legend()
        plt.ylabel(r'$E_z(z)$ (V/m)', fontsize=18)
        plt.xlabel(r'$z$ (arb. offset in m)', fontsize=18)
        plt.ylim(1.2*np.min(-WakePot), 1.2*np.max(-WakePot))
        # plt.xlim(0,max(zz))
        plt.tight_layout()
        plt.savefig(f'figs/wake_a{Inner_Radius*1e3:.2f}mm_delta{Layer_Thickness*1e6:.2f}um.png')
        # Conversion to ASTRA unit V/m/nC:
        plt.xlim(0,10*sigmaz*1e3)
        plt.savefig(f'figs/ZOOM_wake_a{Inner_Radius*1e3:.2f}mm_delta{Layer_Thickness*1e6:.2f}um.png')



    loss = Bunch * WakePot /(charge) # loss function = wake * current profile / charge. wake = V/nC, current profile = nC/m, charge = nC
    # print('Loss', np.max(wLong)/2 *1e-12,"V/pC")
    energy_loss = np.trapz(loss, zz) * (charge)**2 # Energy = int (loss) ds * Q^2
    # print('Energy loss', L * (1e-9)**2 *  energy_loss*1e3 ,' mJ /nC^2')
    if printer:
        print('WakeLength', WakeLength*1e2, 'cm')

        print('Energy loss from wake integral', energy_loss * 1e6 ,' uJ')
        print('Loss factor',LossFactorWake/charge/TubeL * 1e-3 * 1e-12,"kV/(pC m) vs theoretical upper limit", LossFactorCorrugated* 1e-3 * 1e-12,"kV/(pC m)")
        print('Kappa fraction', LossFactorWake/charge/TubeL/LossFactorCorrugated * 100, '%')
#         print('Pulse Length', Pulese_Length*1e12, 'ps')
#         print('Power', power_loss*1e-3, 'kW')

    Pulse_Length = TubeL/vGroup - TubeL/c

    power_loss = energy_loss/Pulse_Length

    # return energy_loss,power_loss, LossFactorWake, LossFactorCorrugated,LossFactor0, WakeLength
    return energy_loss,power_loss, LossFactorWake, LossFactorGreens,LossFactorCorrugated,LossFactor0, WakeLength
#import warnings

#warnings.filterwarnings('error')
def FEL1DTheory(Resonance_Frequency,k0,vPhase, vGroup,gamma,LossFactor,PulseLengthTime, Inner_Radius, Layer_Thickness, TubeL, er, charge, sigmaz,printer=False):
    Resonance_Wavelength = c/Resonance_Frequency

    U_parameter = Inner_Radius*k0/gamma

    UndulatorWaveNumber = k0*(vPhase-vGroup)/vPhase
    UndulatorWaveLength = 2*np.pi/UndulatorWaveNumber

    # Current = charge/PulseLengthTime
    Current = (charge/PulseLengthTime)/np.sqrt(2*np.pi) # peak current
    Current = Current / np.sqrt(2) # RMS current, not the peak current

    n0 = Current/(electron_charge*c) # particles per unit length
    r0 = 1/(4*np.pi*e0) * electron_charge**2 /(m_e * c**2) # classical electron radius
    omega_r = 2*np.pi*Resonance_Frequency
    # kappa0 = 2/(Inner_Radius**2) # limit for gamma to infinity


    # kappa0 =  2/(Inner_Radius**2) # limit for gamma to infinity

    # # # this was in gaussian units, conversion factor is Z0*cLight/(4*np.pi)
    # kappa0 = Z0*c/(4*np.pi)  * kappa0 #  
    # kappa0 = kappa0/charge/TubeL
    # print('Loss factor',  kappa * 1e-3 * 1e-12, 'kV/(pC m) ')

    # kappa = LossFactor/charge/TubeL # This is loss factor per charge per meter in V/C/m instead of just V
    # kappa = LossFactor/charge # This is loss factor per charge in V/C instead of just V
    kappa = LossFactor #

    Alfven_Current = 17.5e3  
    try:
        Gain_Length =   1/3**0.5 * gamma * (  kappa*UndulatorWaveNumber*Current/Alfven_Current /(Z0*c/(4*np.pi)))**(-1/3) 
    except:
        print('Gain length failed')
        Gain_Length = 0
        print('gamma', gamma)
        print('kappa', kappa)
        print('UndulatorWaveNumber', UndulatorWaveNumber)
        print('Current', Current)
        

    # Pierce_Parameter =  0.5 * (  2 * n0 *  kappa * cLight * r0 / ( UndulatorWaveNumber * gamma * omega_r )  )**(1/3)

    try:
        Pierce_Parameter = (1/(4*np.pi*np.sqrt(3))) * UndulatorWaveLength/Gain_Length
    except:
        print('Pierce parameter failed')
        Pierce_Parameter = 0
        print('UndulatorWaveLength', UndulatorWaveLength)
        print('Gain_Length', Gain_Length)


    PSaturation =  Pierce_Parameter * gamma * m_e * c**2 * Current / electron_charge 
    ESaturation = Pierce_Parameter * gamma * m_e * c**2 * charge / electron_charge

    try:
        UndulatorK = np.sqrt( 2*( Resonance_Wavelength/UndulatorWaveLength  * 2 * gamma**2  - 1 )   )
    except:
        print('UndulatorK failed')
        UndulatorK = 0
        print('resonance wavelength', Resonance_Wavelength)
        print('UndulatorWaveLength', UndulatorWaveLength)
        print('gamma', gamma)
        print('vgroup', vGroup/c)
        print('vphase', vPhase/c)
    if printer:
        print('lambda', Resonance_Wavelength*1e3, 'mm')
        print('U parameter', U_parameter)
        
        print('Current at AREAL', Current, 'A')
    
        print('Undulator WaveNumber', UndulatorWaveNumber, '1/m')
        print('Undulator WaveLength', UndulatorWaveLength*1e3, 'mm')
        print('Gain length', Gain_Length*1e2, 'cm')
        print('Saturation length', 20*Gain_Length*1e2, 'cm')
        print('Pierce parameter', Pierce_Parameter  )
        print('P saturation', PSaturation*1e-6, 'MW')

        print('Undulator K', UndulatorK)
    return U_parameter, Gain_Length, PSaturation, ESaturation, kappa, Pierce_Parameter


# if in main, run the following
if __name__ == "__main__":
    # a = 0.1629e-2
    # a = 0.2e-2	
    a = 1000e-6
    # # delta = 0.239469e-2 - a
    delta=200e-6
    delta=10e-9
    # # er = 3.81
    # er = 2
    er = 9.42
    # pz = 4.12 # MeV/c
    pz = 1000 # MeV/c
    gamma = pz/(m_e_MeV)
    vPhase = (1 - 1/gamma**2)**(1/2) * c

    # print('vPhase', vPhase/c ,'c')
    OmegaList = np.linspace(1e9, 50e12, 100000)*2*np.pi

    Resonance_Frequency, k0, freq_0, vGroup = FindWaveguideParam(OmegaList, vPhase, a, delta, er, printer=True)
    print(f'a = {a*1e3:.2f} mm, delta = {delta*1e6:.2f} um, er = {er}')
    print(f"Resonance Frequency: {Resonance_Frequency*1e-9:.2f} GHz")
    # print(f"k0: {k0}")
    # print(f"freq_0: {freq_0*1e-9:.2f} GHz")
    # print(f"vGroup: {vGroup/c:.2f} c")

    a = 250e-6
    a = 1000e-6
    # delta = 0.239469e-2 - a
    # delta=150e-6
    er = 3.81
    # er = 9.42
    # er = 3.81
    # pz = 4.12 # MeV/c
    pz = 10000 # MeV/c
    gamma = pz/(m_e_MeV)
    vPhase = (1 - 1/gamma**2)**(1/2) * c

    print('vPhase', vPhase/c ,'c')
    # OmegaList = np.linspace(1e9, 500e9, 100000)*2*np.pi

    # Resonance_Frequency, k0, freq_0, vGroup = FindWaveguideParam(OmegaList, vPhase, a, delta, er, printer=True)
    OmegaGoal=50.70e9 * 2 * np.pi
    DeltaSamples = np.linspace(1e-9, 1000e-6, 100000)
    Delta = FindDelta(DeltaSamples, a, OmegaGoal, vPhase, er)

    print(f'a = {a*1e3:.2f} mm, delta = {Delta*1e6:.2f} um, er = {er}')

    # print(f"Resonance Frequency: {Resonance_Frequency*1e-9:.2f} GHz")
    # print(f"k0: {k0}")
    # print(f"freq_0: {freq_0*1e-9:.2f} GHz")
    # print(f"vGroup: {vGroup/c:.2f} c")

    print('Test')   