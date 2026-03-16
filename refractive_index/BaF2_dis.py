"""BaF2 disperison data (in terms of wavelength in um) from 
J. Phys. Chem. Ref. Data 9, 161-289 (1980) 
0.15-15 µm
"""

import numpy as np
from numpy.lib.scimath import sqrt
Pi=np.pi
# speed of light in vacuum (m/s)
c = 299792458.0

# vacuum permittivity (F/m)
e0 = 8.854187817e-12

lambda_min=0.15
lambda_max=15

def n(x):  
    return (1+0.33973+0.81070/(1-(0.10065/x)**2)+0.19652/(1-(29.87/x)**2)+4.52469/(1-(53.82/x)**2))**0.5


"""nonlinear dispersion"""

def chi2():
    """returns chi2(:,:,:)
    might be wrong I didnot check"""
    
    chi=np.zeros((3,3,3))

    return 2*chi #2 comes from the historic definition of d

def chi3():
    """returns chi3(:,:,:,:)
    Opt. Lett. 18, 194-196 (1993)
    """
    c_xxxx=1.92*10**-22 #m^2/V^2
    sig=-1.08
    c_xxyy=(1-sig)*c_xxxx/3 #J. Opt. Soc. Am. B 26, 1269-1275 (2009)
    
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

# Chi3=1.92*10**-22
# #https://www.osapublishing.org/josab/abstract.cfm?uri=josab-21-3-640
# lam=2.4
# n2=Chi3/(4/3*n(lam)**2*c*e0)
# print(n2)