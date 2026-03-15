"""CaF2 disperison data (in terms of wavelength in um) from 
# Appl. Opt. 2, 1103-1107 (1963)
# 0.23-9.7 µm

 J. Phys. Chem. Ref. Data 9, 161-289 (1980)
 
"""

import numpy as np
from numpy.lib.scimath import sqrt
Pi=np.pi
# speed of light in vacuum (m/s)
c = 299792458.0

# vacuum permittivity (F/m)
e0 = 8.854187817e-12

# lambda_min=0.23
# lambda_max=9.7

lambda_min=0.15
lambda_max=12

def n(x):  
    # return (1+0.5675888/(1-(0.050263605/x)**2)+0.4710914/(1-(0.1003909/x)**2)+3.8484723/(1-(34.649040/x)**2))**0.5
    return (1+0.33973+0.69913/(1-(0.09374/x)**2)+0.11994/(1-(21.18/x)**2)+4.35181/(1-(38.46/x)**2))**0.5
    # return sqrt((1+0.33973+0.69913/(1-(0.09374/x)**2)+0.11994/(1-(21.18/x)**2)+4.35181/(1-(38.46/x)**2)))
#needs to be something from np

"""nonlinear dispersion"""

def chi2():
    """returns chi2(:,:,:)
    might be wrong I didnot check"""
    
    chi=np.zeros((3,3,3))

    return 2*chi #2 comes from the historic definition of d

def chi3():
    """returns chi3(:,:,:,:)
    https://doi.org/10.1007/10134958_36
    """
    c_xxxx=1.6*10**-22 #m^2/V^2. approximately. might be 2x less
    c_xxyy=0.6*10**-22 #m^2/V^2
    
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

# print(n(2.4))
# Chi3=1.6*10**-22
# #https://www.osapublishing.org/josab/abstract.cfm?uri=josab-21-3-640
# lam=2.4
# n2=Chi3/(4/3*n(lam)**2*c*e0)
# print(n2)