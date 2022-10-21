# -*- coding: utf-8 -*-
"""
author: Nicolas Carbonare
mail: nicocarbonare@gmail
last updated: June 06, 2021
"""

import numpy as np
import pandas as pd


class Ventilation_Decentralized:
    """ 
    Model of a facade-integrated decentralized ventilation system.
    System model is developed following the data from this report: 
        https://www.dibt.de/de/service/zulassungsdownload/detail/z-513-415    
    
    The model calculates the volume flow as a function of the fan speed.
    Additionally, the model calculates:
        - Heat recovery efficiency
        - Supply and exhaust volume flow unbalances due to pressure difference
        - Room ventilation efficiency (source: TO ADD)
    """
    def __init__(self, **kwargs): 
        '''TO DO
        Initial conditions for the model
        Allowed keyword arguments (**kwargs):
    	----------	
        § meas_sup: array of floats
            measured volume flows for different pressure differences at a 
            single fan speed in supply mode, in m^3/h
        § meas_exh: array of floats
            measured volume flows for different pressure differences at a 
            single fan speed in exhaust mode, in m^3/h
        § Vroom: float (default = 60.0)
            room volume where the fan is installed, in m^3
        § bool_hrc: boolean (default = True)
            boolean for calculation of heat recovery in the system
        § bool_noWind: boolean (default = True)
            boolean for calculation of wind pressure effect in the system
        § bool_effVent: boolean (default = True)
            boolean for calculation of ventilation efficiency in the system           
        '''
               
        allowed_keys = {'meas_sup', 'meas_exh', 'Vroom',
                        'bool_hrc','bool_noWind', 'bool_effVent'}

        rejected_keys = set(kwargs.keys()) - set(allowed_keys)
        if rejected_keys:
            raise ValueError("Invalid arguments in constructor: {}".
                             format(rejected_keys))
        else:
            self.default_values()
            self.__dict__.update((k, v) for k, v in kwargs.items() 
            if k in allowed_keys)
            self.check_speeds_dict()
            self.inital_values()
          
           
    def default_values(self):
        '''
        Reset the values to the default ones.
        These values do not change during simulation.
        '''
        self.meas_sup = ({
            1000: [38,34,28,23,17,7], #rpm = 1000
            2000: [48,45,41,36,31,25,18], #rpm = 2000
            2750: [60,57,52,50,46,40,35,30,24] #rpm = 2750
            })
        self.meas_exh = ({
            1000: [41,36,30,23,17,10], #rpm = 1000
            2000: [48,44,41,36,32,27,23], #rpm = 2000
            2750: [61,57,54,49,46,41,37,32,26] #rpm = 2750
            })
        self.dp_array = np.arange(-20,25,5)*-1
        
        self.Vroom = 60
        
        # Boolean flags for calculations
        self.bool_hrc = True 
        self.bool_noWind = False 
        self.bool_effVent = False


    def check_speeds_dict(self):
        '''
        Check consistency of manually-written speeds array
        '''
        if self.meas_sup.keys() != self.meas_exh.keys():
            raise ValueError("Speeds for supply and exhaust not matching")


    def inital_values(self):
        '''
        Initialize the values required for calculation
        These values can change in the next step, the 
        ones in self.default_values not. 
        '''
        Nrpm_array = list(self.meas_sup.keys())
        
        self.VF_sup_eff = 1
        self.VF_exh_eff = 1
        self.hrc = 0
        
        self.Nrpm_max = max(Nrpm_array)
        self.Nrpm_min = min(Nrpm_array)
       
        
    def heatrecovery(self, vflow):
        '''
        Calculates the heat recovery efficiency as a function of the volume flow
        Exponential efficiency curve
        Parameters:
    	----------	
        § vflow: float
            volume flow, in m^3/h
        Returns:
        -------
        § hrc: float
            heat recovery efficiency, in % (fraction)
        '''      

        hrc_eff = 1.00112547 * np.exp(-0.01598704 * vflow) + 0.01691251
        
        return hrc_eff


    def efficiency_ventilation(self, VF_sup, VF_exh, Vroom):
        '''
        Calculates the ventilation efficiency in a certain room volume.
        Parameters:
    	----------	
        § VF_sup: float
            Volume flow in supply phase, in m^3/h
        § VF_exh: float
            Volume flow in exhaust phase, in m^3/h
        § Vroom: float
            Room volume, in m^3
        Returns:
        -------
        § VF_sup_eff: float
            Ventilation efficiency in supply phase, in % (fraction)
        § VF_exh_eff: float
            Ventilation efficiency in exhaust phase, in % (fraction)
        '''
            
        VF_sup_eff = -0.244*(VF_sup/Vroom)**2 + 0.376*(VF_sup/Vroom) + 0.732
        VF_exh_eff = -0.244*(VF_exh/Vroom)**2 + 0.376*(VF_exh/Vroom) + 0.732
            
        return [VF_sup_eff, VF_exh_eff]


    def calc_vflow(self, Nrpm, dP=0):
        '''
        Calculate the volume flow as a function of the fan speed. 
        Fit polynomial approximations of resulting volume flow as a function 
        of the pressure difference due to wind for every fan level.
        Parameters:
    	----------	
        § Nrpm: float
            Fan speed, in RPM
        § dP: float
            Pressure difference between indoor and outdoor environment, in Pa
        Returns:
        -------
        § VF_sup: float
            Volume flow in supply phase, in m^3/s
        § VF_exh: float
            Volume flow in supply phase, in m^3/s
        '''
        Nrpm = min(Nrpm, self.Nrpm_max)
        Nrpm = max(Nrpm, 0)

        lev = sum([Nrpm > i for i in list(self.meas_sup.keys())])
        lev_up = list(self.meas_sup.keys())[lev]
        lev_diff = lev_up
        yp_poly_sup_up = np.poly1d(np.polyfit(self.dp_array[:len(self.meas_sup[lev_up])]*-1, 
                                            self.meas_sup[lev_up], 2))
        yp_poly_exh_up = np.poly1d(np.polyfit(self.dp_array[:len(self.meas_exh[lev_up])]*-1, 
                                            self.meas_exh[lev_up], 2))
                                            
        if lev > 0:
            lev_lo = list(self.meas_sup.keys())[lev-1]
            lev_diff = lev_up - lev_lo
            yp_poly_sup_lo = np.poly1d(np.polyfit(self.dp_array[:len(self.meas_sup[lev_lo])]*-1, 
                                            self.meas_sup[lev_lo], 2))
            yp_poly_exh_lo = np.poly1d(np.polyfit(self.dp_array[:len(self.meas_exh[lev_lo])]*-1, 
                                            self.meas_exh[lev_lo], 2))
        
        if Nrpm <= self.Nrpm_min:
            VF_sup = yp_poly_sup_up(dP)/3600*(Nrpm/self.Nrpm_min)   
            VF_exh = yp_poly_exh_up(dP)/3600*(Nrpm/self.Nrpm_min)   
        else:
            VF_sup = ((yp_poly_sup_up(dP) - yp_poly_sup_lo(dP))*((Nrpm-lev_lo)/lev_diff)+yp_poly_sup_lo(dP))/3600       
            VF_exh = ((yp_poly_exh_up(dP) - yp_poly_exh_lo(dP))*((Nrpm-lev_lo)/lev_diff)+yp_poly_exh_lo(dP))/3600  
        
        return [VF_sup, VF_exh]
        
               
    def output(self, Nrpm, dP=0, pr=True):
        '''
        Function to calculate the volume flow and heat recovery efficiency
        as a function of the fan speed and pressure difference between rooms.
        Parameters:
    	----------	
        § Nrpm: integer
            Fan speed, in RPM
        § dP: float
            Pressure difference between indoor and outdoor environment, in Pa
        Returns:
        -------
        § VF_sup: float
            Volume flow in supply phase, in m^3/s
        § VF_exh: float
            Volume flow in supply phase, in m^3/s
        § hrc_Eff: float
            Heat recovery efficiency in supply phase, in %
        '''

        VF_sup, VF_exh = self.calc_vflow(Nrpm, dP)
                            
        if self.bool_effVent == True:
            VF_sup, VF_exh = self.efficiency_ventilation(VF_sup*3600, VF_exh*3600, self.Vroom)

        if self.bool_hrc == True: 
            self.hrc_eff = self.heatrecovery((VF_sup+VF_exh)*3600/2) #convert to m^3/h

        if self.bool_noWind == True:
            VF = (VF_sup+VF_exh)/2 
            VF_sup, VF_exh = VF, VF
            
        self.VF_sup = VF_sup
        self.VF_exh = VF_exh

        return [self.VF_sup, self.VF_exh, self.hrc_eff]



if __name__ == '__main__':
    pass