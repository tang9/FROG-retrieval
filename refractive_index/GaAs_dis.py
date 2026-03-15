"""GaAs disperison data (in terms of wavelength in um) from 
J. Appl. Phys., 94, 6447-6455 (2003)
0.97-17 µm
"""

from numpy.lib.scimath import sqrt
import numpy as np
Pi=np.pi
# speed of light in vacuum (m/s)
c = 299792458.0

# vacuum permittivity (F/m)
e0 = 8.854187817e-12

lambda_min=0.97
lambda_max=17

def n(x):  
    return (1+4.372514+5.466742/(1-(0.4431307/x)**2)+
            0.02429960/(1-(0.8746453/x)**2)+1.957522/(1-(36.9166/x)**2))**.5
    
"""nonlinear dispersion"""
#D:\MPQ\liter\classical optics\nonlinear optics\3-d order nonlinearities\n2 solids\GaAs.pdf

def chi2():
    """returns chi2(:,:,:)
    might be wrong I didnot check"""
    
    chi=np.zeros((3,3,3))
    
    #d=1.8*10**-10 #m/V for SHG

    return 2*chi #2 comes from the historic definition of d

def chi3():
    """returns chi3(:,:,:,:)
    """
    c_xxxx=0.97*10**-11*4*Pi/(10**-4*c)**(3-1) #m^2/V^2 including conversion from esu to SI
    c_xxyy=0.51*10**-11*4*Pi/(10**-4*c)**(3-1)
    #https://books.google.com/books?id=su_pUXQqm5sC&pg=PA296&lpg=PA296&dq=esu+si+third+susceptibility&source=bl&ots=VD14ROw-8p&sig=ACfU3U3MjqPhC7gTAva9Ie4VyKXOT5P5wQ&hl=ru&sa=X&ved=2ahUKEwiRlKzc263jAhWDGs0KHTxjDrUQ6AEwAnoECAYQAQ#v=onepage&q=esu%20si%20third%20susceptibility&f=false
    
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