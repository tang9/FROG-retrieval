"""LiF disperison data (in terms of wavelength in um) from 
J. Phys. Chem. Ref. Data 5, 329-528 (1976) 
0.1-11 µm
"""

from numpy.lib.scimath import sqrt
import numpy as np
# speed of light in vacuum (m/s)
c = 299792458.0

# vacuum permittivity (F/m)
e0 = 8.854187817e-12

lambda_min=0.1
lambda_max=11

def n(x):  
    return sqrt(1+0.92549/(1-(0.07376/x)**2)+6.96747/(1-(32.790/x)**2))

"""nonlinear dispersion"""

def chi2():
    """returns chi2(:,:,:)
    might be wrong I didnot check"""
    
    chi=np.zeros((3,3,3))

    return 2*chi #2 comes from the historic definition of d

def chi3(lam=1):
    """returns chi3(:,:,:,:)
    Phys. Rev. B 4, 3437 – Published 15 November 1971  //  Appl. Phys. Lett. 31, 822 (1977)  // 898 Sov. Phys. JETP 46(5), Nov. 1977 
    """
    
    n2=1.2*10**-20 #m^2/W nonlinear refrective index.
    c_xxxx=n2*4/3*c*e0*n(lam)**2 #m^2/V^2
    c_xxyy=0.45*c_xxxx
    
    chi=np.zeros((3,3,3,3))
    
    #since it is a cubic crystal
    chi[0][0][0][0]=c_xxxx
    chi[1][1][1][1]=c_xxxx
    chi[2][2][2][2]=c_xxxx
    
    chi[0][0][1][1]=c_xxyy
    chi[0][1][0][1]=c_xxyy
    chi[0][1][1][0]=c_xxyy
    chi[0][0][2][2]=c_xxyy
    chi[0][2][0][2]=c_xxyy
    chi[0][2][2][0]=c_xxyy
    chi[1][1][0][0]=c_xxyy
    chi[1][0][1][0]=c_xxyy
    chi[1][0][0][1]=c_xxyy
    chi[1][1][2][2]=c_xxyy
    chi[1][2][1][2]=c_xxyy
    chi[1][2][2][1]=c_xxyy
    chi[2][2][0][0]=c_xxyy
    chi[2][0][2][0]=c_xxyy
    chi[2][0][0][2]=c_xxyy
    chi[2][2][1][1]=c_xxyy
    chi[2][1][2][1]=c_xxyy
    chi[2][1][1][2]=c_xxyy
    
    return chi