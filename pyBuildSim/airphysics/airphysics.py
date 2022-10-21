# -*- coding: utf-8 -*-
"""
authors: Felix Ohr, Nicolas Carbonare
mail: fohr@ise.fhg.de, ncarbona@gmail.com
last updated: March 13, 2019
"""
import matplotlib.pylab as pylab

'''
thermodynamics for humid air
'''
#international barometric altitude equation for international standard conditions for the atmosphere
#t = 15 °C (288.15 K), p0 = 101325.0 Pa
T0                                      = 288.15        # standard temperature in K
p0                                      = 101325.0      # standard pressure in Pa
r0                                      = 2502000.0     # specific enthalpie of water vaporization in kJ/kg
cp_da                                   = 1008.0        # specific heat capacity of dry air in kJ/kgK
cp_wv                                   = 1860.0        # specific heat capacity of water vapor in kJ/kgK
cp_w                                    = 4186.0        # specific heat capacity of liquid water in kJ/kgK
R_da                                    = 287.055       # ideal gas constant for dry air in J/kgK
M_da                                    = 28.9645       # molar mass of dry air in kg/kmol 
M_wv                                    = 18.01528      # molar mass of water vapour in kg/kmol 


def calc_p_sat_ASHRAE(t):
    '''
    INPUT: TEMPERATURE IN DEGREE C    
    SUBROUTINE FOR FINDING SATURATION PRESSURE OF WATER AT A GIVEN
    TEMPERATURE TAKEN ASHRAE
    '''
    #Constants for calculation of p_sat according ASHRAE equation
    C1                                  = -5.674359e3
    C2                                  =  6.3925247
    C3                                  = -9.677843e-3
    C4                                  =  6.2215701e-7
    C5                                  =  2.0747825e-9
    C6                                  = -9.484024e-13
    C7                                  =  4.1635019
    C8                                  = -5.8002206e3
    C9                                  =  1.3914993
    C10                                 = -4.8640239e-2
    C11                                 =  4.1764768e-5
    C12                                 = -1.4452093e-8
    C13                                 =  6.5459673
    
    T                                   = t + 273.15
    p_sat                               = pylab.ones_like(T)*pylab.nan
    index1                              = (T >173.15) & (T< 273.15)
    index2                              = (T>=273.15) & (T<=473.15)
    p_sat[index1]                       = pylab.exp(C1/T[index1] + C2 + \
                                                    C3*T[index1] + \
                                                    C4*T[index1]**2 + \
                                                    C5*T[index1]**3 + \
                                                    C6*T[index1]**4 + \
                                                    C7*pylab.log(T[index1]))

    p_sat[index2]                       = pylab.exp(C8/T[index2] + C9 + \
                                                    C10*T[index2] + \
                                                    C11*T[index2]**2 + \
                                                    C12*T[index2]**3 + \
                                                    C13*pylab.log(T[index2]))
    return p_sat


def calc_p_sat_ASHRAE_float(t):
    '''
    INPUT: TEMPERATURE IN DEGREE C    
    SUBROUTINE FOR FINDING SATURATION PRESSURE OF WATER AT A GIVEN
    TEMPERATURE TAKEN ASHRAE
    Function from @ncarbona
    '''
    #Constants for calculation of p_sat according ASHRAE equation
    C8                                  = -5.8002206e3
    C9                                  =  1.3914993
    C10                                 = -4.8640239e-2
    C11                                 =  4.1764768e-5
    C12                                 = -1.4452093e-8
    C13                                 =  6.5459673
    
    T                                   = t + 273.15
    p_sat                               = pylab.exp(C8/T + C9 + \
                                                    C10*T + \
                                                    C11*T**2 + \
                                                    C12*T**3 + \
                                                    C13*pylab.log(T))
    return p_sat
#equation to calculate water steam content in dry air
#derived from ideal gas equation
'''
security for impossible input values has to be inserted
'''
def calc_x(relHum, p_sat, pLoc):
    x = M_wv/M_da * relHum*p_sat/(pLoc - relHum*p_sat)
    return x

#equation to calculate specific enthalpie of moist and dry air, without condensation
#for dry air, x has to be set zero
'''
security for impossible input values has to be inserted
'''
def calc_specEnthalpie(t,x):
    ent = cp_da*t + x*(r0 + cp_wv*t)
    return ent


#barometric equation to calc the pressure depending on height over sea level of location
'''
security for impossible input values has to be inserted
'''
def calc_p(h):
    p=p0*(1 - (0.0065 * h)/T0)**5.255
    return p
    

def calc_relHum(x,p_sat,pLoc):
    relHum = x*pLoc/(p_sat*(M_wv/M_da + x))
    return relHum


if __name__ == '__main__':
    pass
#    '''
#    check vapor saturation pressure curve
#    '''
#    t           = pylab.arange(0,100,1)
#    p_sat       = calc_p_sat_ASHRAE(t)
#    
#    # values from VDI Waermeatlas
#    t_vdi       = pylab.array([0.,
#                               5.,
#                               10.,
#                               15.,
#                               20.,
#                               25.,
#                               30.,
#                               35.,
#                               40.,
#                               45.,
#                               50.,
#                               55.,
#                               60.,
#                               65.,
#                               70.,
#                               75.,
#                               80.,
#                               85.,
#                               90.,
#                               95.,
#                               100.])
#    
#    p_sat_vdi   = pylab.array([0.006112,
#                               0.008726,
#                               0.012282,
#                               0.017057,
#                               0.023392,
#                               0.031697,
#                               0.042467,
#                               0.056286,
#                               0.073844,
#                               0.095944,
#                               0.12351,
#                               0.15761,
#                               0.19946,
#                               0.25041,
#                               0.31201,
#                               0.38595,
#                               0.47415,
#                               0.57867,
#                               0.70182,
#                               0.84609,
#                               1.01420])
#    
#    pylab.figure(figsize=[12,9], facecolor='w', edgecolor='w')
#    pylab.plot(t, p_sat/10**5, 'k-', label='berechnete Werte nach ASHRAE', linewidth=3)
#    pylab.plot(t_vdi, p_sat_vdi, 'bx', label='Messwerte nach VDI Waermeatlas', markersize=12)
#    pylab.title('Validierung Saettigungsdruckkurve',fontsize=16)
#    pylab.xlabel('Temperatur in degC',fontsize=14)
#    pylab.ylabel('Saettigungsdruck Wasserdampf in bar',fontsize=14)
#    pylab.grid()
#    pylab.tight_layout()
#    pylab.legend(loc=0,fontsize=14)
#    pylab.show()
#    