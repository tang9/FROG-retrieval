""""fused silica disperison data (in terms of wavelength in um) from 
https://refractiveindex.info/?shelf=glass&book=fused_silica&page=Malitson
Malitson 1965: n 0.21-3.71 µm

n2
@1030nm
Opt. Express 27, 11018-11028 (2019)
Appl. Opt. 54, F123-F128 (2015)"""

import numpy as np
# speed of light in vacuum (m/s)
c = 299792458.0

# vacuum permittivity (F/m)
e0 = 8.854187817e-12

lambda_min=0.21
lambda_max=3.71

def n(lam):
    C1 = 0.6961663
    C2 = 0.0684043
    C3 = 0.4079426
    C4 = 0.1162414
    C5 = 0.8974794
    C6 = 9.896161
    return (1. + C1*lam**2/(lam**2 - C2**2) + C3*lam**2/(lam**2 - C4**2) + C5*lam**2/(lam**2 - C6**2))**0.5

"""nonlinear dispersion"""

def chi2():
    """returns chi2(:,:,:)
    might be wrong I didnot check"""
    
    chi=np.zeros((3,3,3))

    return 2*chi #2 comes from the historic definition of d

def chi3(lam=1):
    """returns chi3(:,:,:,:)
    """
    n2=2.2*10**-20 #m^2/W nonlinear refrective index. difficult value because the data is spread between 0.5 and 3

    c_xxxx=n2*4/3*c*e0*n(lam)**2 #m^2/V^2
    
    chi=np.zeros((3,3,3,3))
    
    chi[0][0][0][0]=c_xxxx
    chi[1][1][1][1]=c_xxxx
    chi[2][2][2][2]=c_xxxx
    
    return chi