"""Al2O3 (sapphire) disperison data (in terms of wavelength in um) from 
J. Opt. Soc. Am. 62, 1405 (1972)
0.2-5 µm"""

import numpy as np
from numpy.lib.scimath import sqrt
Pi=np.pi
import numpy as np
# speed of light in vacuum (m/s)
c = 299792458.0

# vacuum permittivity (F/m)
e0 = 8.854187817e-12

lambda_min=0.2
lambda_max=5

def n(x):
    # no=sqrt(1+1.4313493/(1-(0.0726631/x)**2)+0.65054713/(1-(0.1193242/x)**2)+5.3414021/(1-(18.028251/x)**2))
    # ne=sqrt(1+1.5039759/(1-(0.0740288/x)**2)+0.55069141/(1-(0.1216529/x)**2)+6.5927379/(1-(20.072248/x)**2))
    no=(1+1.4313493/(1-(0.0726631/x)**2)+0.65054713/(1-(0.1193242/x)**2)+5.3414021/(1-(18.028251/x)**2))**0.5
    ne=(1+1.5039759/(1-(0.0740288/x)**2)+0.55069141/(1-(0.1216529/x)**2)+6.5927379/(1-(20.072248/x)**2))**0.5
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
    n2=2.7*10**-20 #m^2/W nonlinear refrective index. J Appl Glass Sci. 2018; 9: 421‐ 427

    #it is not really correct because there has to be some dependence for a crystal but I have not found it
    c_xxxx=n2*4/3*c*e0*n(lam)**2 #m^2/V^2
    
    chi=np.zeros((3,3,3,3))
    
    chi[0][0][0][0]=c_xxxx
    chi[1][1][1][1]=c_xxxx
    chi[2][2][2][2]=c_xxxx
    
    return chi