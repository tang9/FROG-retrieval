"""SiO2 disperison data (in terms of wavelength in um) from 
Opt. Commun. 163, 95-102 (1999)
0.2-2 µm"""

import numpy as np
from numpy.lib.scimath import sqrt
Pi=np.pi
import numpy as np
# speed of light in vacuum (m/s)
c = 299792458.0

# vacuum permittivity (F/m)
e0 = 8.854187817e-12

lambda_min=0.2
lambda_max=2

def n(x):
    no=sqrt(1+0.28604141+1.07044083/(1-1.00585997e-2/x**2)+1.10202242/(1-100/x**2))
    ne=sqrt(1+0.28851804+1.09509924/(1-1.02101864e-2/x**2)+1.15662475/(1-100/x**2))
    return np.array([no,ne])

"""nonlinear dispersion"""

def chi2():
    """returns chi2(:,:,:)
    might be wrong I didnot check"""
    
    chi=np.zeros((3,3,3))

    return 2*chi #2 comes from the historic definition of d

def chi3(lam=1):
    """returns chi3(:,:,:,:)
    """
    n2=1.5*10**-20 #m^2/W nonlinear refrective index. ?

    c_xxxx=n2*4/3*c*e0*n(lam)**2 #m^2/V^2
    
    chi=np.zeros((3,3,3,3))
    
    chi[0][0][0][0]=c_xxxx
    chi[1][1][1][1]=c_xxxx
    chi[2][2][2][2]=c_xxxx
    
    return chi