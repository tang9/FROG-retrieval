"""ref index info
Li 1976
"""

from numpy.lib.scimath import sqrt
import numpy as np
# speed of light in vacuum (m/s)
c = 299792458.0

# vacuum permittivity (F/m)
e0 = 8.854187817e-12

lambda_min=0.2
lambda_max=42

def n(x):  
    return (1+0.39408+0.79221/(1-(0.146/x)**2)+0.01981/(1-(0.173/x)**2)+
            0.15587/(1-(0.187/x)**2)+0.17673/(1-(60.61/x)**2)+2.06217/(1-(87.72/x)**2))**.5


"""nonlinear dispersion"""

def chi2():
    """returns chi2(:,:,:)
    should be wrong needs to be found"""
    
    chi=np.zeros((3,3,3))

    return 2*chi #2 comes from the historic definition of d

def chi3(lam=1):
    """returns chi3(:,:,:,:)
    """
    n2=0 #m^2/W nonlinear refrective index.
    #to do take dependance [doi: 10.1117/12.628686 Proc. of SPIE Vol. 5971 59710H-2, Opt. Express 22, 5852-5858 (2014)]

    c_xxxx=n2*4/3*c*e0*n(lam)**2 #m^2/V^2
    
    chi=np.zeros((3,3,3,3))
    
    chi[0][0][0][0]=c_xxxx
    chi[1][1][1][1]=c_xxxx
    chi[2][2][2][2]=c_xxxx
    
    return chi