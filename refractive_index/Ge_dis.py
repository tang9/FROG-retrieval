"""Ge disperison data (in terms of wavelength in um) from 
Proc. SPIE 9974, 99740X (2016)
2-14 µm
"""

from numpy.lib.scimath import sqrt
import numpy as np
# speed of light in vacuum (m/s)
c = 299792458.0

# vacuum permittivity (F/m)
e0 = 8.854187817e-12

lambda_min=2
lambda_max=14

def n(x):  
    # return sqrt(1+0.4886331/(1-1.393959/x**2)+14.5142535/(1-0.1626427/x**2)+0.0091224/(1-752.190/x**2))
    return (1+0.4886331/(1-1.393959/x**2)+14.5142535/(1-0.1626427/x**2)+0.0091224/(1-752.190/x**2))**.5
#(1+0.4886331/(1-1.393959/x**2)+14.5142535/(1-0.1626427/x**2)+0.0091224/(1-752.190/x**2))**2

"""nonlinear dispersion"""

def chi2():
    """returns chi2(:,:,:)
    might be wrong I didnot check"""
    
    chi=np.zeros((3,3,3))

    return 2*chi #2 comes from the historic definition of d

def chi3(lam=3.5):
    """returns chi3(:,:,:,:)
    """
    n2=30*10**-18 #m^2/W nonlinear refrective index.@3.5um Nanophotonics 2014; 3(4-5) 247–268

    c_xxxx=n2*4/3*c*e0*n(lam)**2 #m^2/V^2
    
    chi=np.zeros((3,3,3,3))
    
    chi[0][0][0][0]=c_xxxx
    chi[1][1][1][1]=c_xxxx
    chi[2][2][2][2]=c_xxxx
    
    return chi