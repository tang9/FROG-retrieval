"""Air simple disperison data (in terms of wavelength in um) from 
Appl. Optics 35, 1566-1573 (1996)
dry air at 15 °C, 101.325 kPa and with 450 ppm CO2 content
"""

import numpy as np
from numpy.lib.scimath import sqrt
Pi=np.pi
# speed of light in vacuum (m/s)
c = 299792458.0

# vacuum permittivity (F/m)
e0 = 8.854187817e-12


lambda_min=0.23
lambda_max=1.69

def n(x):  
    return 1+0.05792105/(238.0185-x**-2)+0.00167917/(57.362-x**-2)

"""nonlinear dispersion"""

def chi2():
    """returns chi2(:,:,:)"""
    
    chi=np.zeros((3,3,3))

    return 2*chi #2 comes from the historic definition of d

def chi3(lam=1,LongPulse=True):
    """returns chi3(:,:,:,:)
    long pulse is >~150fs"""
    
    #m^2/W nonlinear refrective index. Opt. Lett. 40, 5794-5797 (2015)
    n2_N2=8*10**-24 #m^2/W
    n2rot_N2=24*10**-24 #rotatinal component
    n2_O2=9*10**-24 #m^2/W
    n2rot_O2=54*10**-24 #rotatinal component
    
    if LongPulse:
        n2=0.8*(n2_N2+n2rot_N2)+0.2*(n2_O2+n2rot_O2)
    else:
        n2=0.8*n2_N2+0.2*n2_O2

    c_xxxx=n2*4/3*c*e0*n(lam)**2 #m^2/V^2
    
    chi=np.zeros((3,3,3,3))
    
    chi[0][0][0][0]=c_xxxx
    chi[1][1][1][1]=c_xxxx
    chi[2][2][2][2]=c_xxxx
    
    return chi