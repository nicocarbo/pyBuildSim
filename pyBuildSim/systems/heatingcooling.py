# -*- coding: utf-8 -*-

"""
author: Nicolas Carbonare
mail: nicocarbonare@gmail
last updated: June 06, 2021
"""
import numpy as np
import pandas as pd

class Boiler:
    """ 
    Characteristic model of a gas boiler for heating and DHW systems.
    Boiler efficiency is calculated according to the following methods:     
    - PLR calculation method -> doi.org/10.1007/s11708-018-0596-5
    - T_in calculation method -> Sigmoid function - TO DO
    - PLR + T_in calculation method -> doi.org/10.1016/j.enconman.2017.01.016
    
    The model returns both useful heat and gas mass flow consumed. 
    """
    def __init__(self, **kwargs): 
        '''
        Initial conditions for the model
        Allowed keyword arguments (**kwargs):
    	----------	
        § Qflow_nom: float (default = 15000.0)
            nominal thermal power of the boiler in W
        § eta_nom: float (default = 0.9)
            nominal thermal efficiency of the gas combustion process, in fraction
        § cp: float (default = 4180.0)
            specific heat capacity secondary circuit, kJ/kgK
        § min_plr: float (default = 0.1)
            mininum partload fraction of calculated heat pump, in fraction
        § eff: string (default = 'const')
            calculation method of the efficiency curve              
        '''
               
        allowed_keys = {'Qflow_nom', 'eta_nom', 'cp',
                        'min_plr','eff',
                        'a','b','c','d'}

        rejected_keys = set(kwargs.keys()) - set(allowed_keys)
        if rejected_keys:
            raise ValueError("Invalid arguments in constructor: {}".
                             format(rejected_keys))
        else:
            self.default_values()
            self.__dict__.update((k, v) for k, v in kwargs.items() 
            if k in allowed_keys)
            self.check_method()
            self.default_curve()
           
           
    def default_values(self):
        '''
        Reset the values to the default ones
        '''
        self.Qflow_nom = 15000.0 
        self.eta_nom = 0.9 
        self.cp = 4180.0 
        self.min_plr = 0.1
        self.eff = 'const'
        
        self.a = 0.0
        self.b = 0.0
        self.c = 0.0
        self.d = 0.0
            
    def default_curve(self):
        '''
        Check if the boiler efficiency curve coefficients were
        defined manually. Otherwise define default curve coefficients.
        Parameters:
    	----------	           
        § a: float 
            coefficient "a" for boiler efficiency curve       
        § b: float 
            coefficient "b" for boiler efficiency curve       
        § c: float
            coefficient "c" for boiler efficiency curve       
        § d: float
            coefficient "d" for boiler efficiency curve          
        '''
        if self.a+self.b+self.c+self.d == 0: 
            if self.eff == 'plr':
                self.a = 0.3822
                self.b = 2.2013
                self.c = -2.8237
                self.d = 1.3021
            elif self.eff == 't_in':
                self.a = 0.065
                self.b = -0.078
                self.c = -43.318
                self.d = 0.905
            else: 
                self.a = 0.0
                self.b = 0.0
                self.c = 0.0
                self.d = 0.0
        else:
            pass              


    def check_method(self):
        '''
        Check of the selected calculation method is defined in the model
        Otherwise raise error        
        Allowed methods:
        § plr: polynomical cubic approximation using the part load ratio
        § t_in: sigmoid function using the return water temperature
        § plr_tin: function based on the part load ratio nad return water temperature
        § const: constant efficiency, assuming the nominal value        
        '''
        allowed_method = {'plr','t_in','plr_tin', 'const'}
        
        if self.eff not in allowed_method:
            raise ValueError("Invalid boiler efficiency calculation method: {}".
                             format(self.eff))
        else:
            pass
        
        
    def eta_plr(self, plr):
        '''
        Calculates the boiler efficiency - eta
        Cubic efficiency curve
        Parameters:
    	----------	
        § plr: float
            part load ratio, in fraction
        Returns:
        -------
        § eta: float
            efficiency of the combustion process
        '''      
        eta = self.a + self.b*plr + self.c*plr**2 + self.d*plr**3
        eta = np.min(np.max(eta,0),1)
        return eta


    def eta_tin(self, T_in):
        '''
        Calculates the boiler efficiency - eta
        Sigmoid function of the efficiency
        Parameters:
    	----------	
        § T_in: float
            return temperature of the fuel flow, in Kelvin
        Returns:
        -------
        § eta: float
            efficiency of the combustion process
        '''
        eta = self.a*np.tanh(self.b*(T_in+self.c)) + self.d
        eta = np.min(np.max(eta,0),1)
        return eta


    def eta_plr_tin(self, plr, T_in):
        '''
        Calculates the boiler efficiency - eta
        Piecewise function using part load ratio and return water
        temperature as inputs
        Parameters:
    	----------
        § plr: float
            part load ratio, in fraction       
        § T_in: float
            return temperature of the fuel flow, in Kelvin
        Returns:
        -------
        § eta: float
            efficiency of the combustion process
        '''
        eta = 7.147 + 8.841*T_in - 0.2564*T_in**2 + 0.3945E-3*T_in**3 - \
            1.7983E-5*T_in**4 + 12.024*plr + 0.271*plr*T_in + 1.23E-3*plr*T_in**2
        eta = np.min(np.max(eta,0),1)
        return eta
        
        
    def output(self, T_out_set, T_in, mflow_in, ctrl, pr=True):
        '''
        Calculates the boiler rated power as a function of different values
        Sigmoid function of the efficiency
        Parameters:
    	----------	
        § T_out_set: float
            temperature set point of the supply water, in Kelvin
        § T_in: float
            return temperature of the fuel flow, in Kelvin
        § mflow_in: float
            water mass flow rate, in kg/s
        § ctrl: integer
            boiler control flag, 0 = off, 1 = on
        Returns:
        -------
        § Qflow_fuel: float
            consumed gas mass flow, in kg/s
        '''
        self.T_in = T_in - 273.15 # inlet fluid temperature, calculation in °C
        self.mflow_in = mflow_in # inlet fluid flow rate
        self.ctrl = ctrl # control signal
        self.T_out_set = T_out_set - 273.15 # requested flow temperature, calculation in °C

        if self.mflow_in <= 0. or self.ctrl <= 0:
            # Boiler is OFF
            self.Qflow_heat_req = 0.0
            self.Qflow_heat = 0.0
            self.T_out = self.T_in + 273.15
            self.plr = 0.0
            self.eta = 0.0
            self.Qflow_fuel = 0.0

        else:
            # Boiler is ON
            # Calculation of requested heat (load)
            self.Qflow_heat_req = self.mflow_in * self.cp * (self.T_out_set - self.T_in)

            if self.Qflow_heat_req > self.Qflow_nom:
                self.Qflow_heat = self.Qflow_nom # delivered thermal power in W
                self.T_out = (self.T_in + (self.Qflow_heat/(self.mflow_in*self.cp))) + 273.15
                self.plr = self.Qflow_heat/self.Qflow_nom # calculation of part load ratio
            else:
                self.Qflow_heat = self.Qflow_heat_req # delivered thermal power in W
                self.T_out = (self.T_in + (self.Qflow_heat/(self.mflow_in*self.cp))) + 273.15
                self.plr = self.Qflow_heat/self.Qflow_nom # calculation of part load ratio

            if self.plr < self.min_plr:
               self.Qflow_heat = 0.0 # delivered thermal power in W
               self.T_out = self.T_in + 273.15
               self.plr = 0.0 # calculation of part load ratio

            # eta calculation
            if self.eff == 'plr':
                self.eta = self.eta_plr(self.plr)
            elif self.eff == 't_in':
                self.eta = self.eta_tin(self.T_in)               
            elif self.eff == 'plr_tin':
                self.eta = self.eta_plr_tin(self.plr, self.T_in)
            else:
                self.eta = self.eta_nom
               
            # Calculation of gas consumption
            self.Qflow_fuel = self.Qflow_heat/self.eta # energy flow rate of fuel, calculation based on gross calorific value (GCV) = upper heating value (UHV)
        
        if pr:
            return [self.Qflow_heat, self.Qflow_fuel]


class HeatPump_effCurve:
    """
    Model of a heat pump using a characteristic curve fit, for space heating/cooling and DHW.
    
    PLR calculation method -> doi.org/10.1007/s11708-018-0596-5
    temp_mass calculation method -> https://hvac.okstate.edu/sites/default/files/pubs/theses/MS/27-Tang_Thesis_05.pdf
    PLR + T_in calculation method -> doi.org/10.1016/j.enconman.2017.01.016
    """
    def __init__(self, **kwargs): 
        '''
        Initial conditions for the model
        Allowed keyword arguments (**kwargs):
    	----------	
        § QLoa_hea_nom: float
            nominal thermal power of the boiler in W
        § QLoa_coo_nom: float
            nominal thermal efficiency of the gas combustion process, in fraction
        § Pcoo_nom: float
            specific heat capacity secondary circuit, kJ/kgK
        § Phea_nom: float
            mininum partload fraction of calculated heat pump, in fraction
        § rev: boolean (default = True)
            flag for reversible heat pumps              
        § cpLoa: float
            specific heat capacity load circuit, kJ/kgK
        § cpSou: float
            specific heat capacity source circuit, kJ/kgK
        § min_plr: float (default = 0.1)
            mininum partload fraction of calculated heat pump, in fraction
        § mLoa_flow_nom: float
            nominal mass flow load circuit, kg/s
        § mSou_flow_nom: float
            nominal mass flow source circuit, kg/s
        § TLoa_coo_nom: float
            nominal outlet temperature load circuit in cooling mode, in Kelvin
        § TSou_coo_nom: float
            nominal outlet temperature source circuit in cooling mode, in Kelvin
        § TLoa_hea_nom: float
            nominal outlet temperature load circuit in heating mode, in Kelvin
        § TSou_hea_nom: float
            nominal outlet temperature source circuit in heating mode, in Kelvin
        § coeQ_coo: array of floats
            cooling thermal power efficiency curve coefficients   
        § coeQ_hea: array of floats
            heating thermal power efficiency curve coefficients   
        § coeP_coo: array of floats
            cooling electrical power efficiency curve coefficients   
        § coeP_hea: array of floats
            heating electrical power efficiency curve coefficients   
        § calc_method: string (default = 'temp')
            calculation method of the efficiency curve     
        '''
                                         
        allowed_keys = {'QLoa_hea_nom', 'QLoa_coo_nom','Pcoo_nom', 'Phea_nom', 'rev',
                        'cpLoa', 'cpSou', 'min_plr','mLoa_flow_nom','mSou_flow_nom',
                        'TLoa_coo_nom','TSou_coo_nom','TLoa_hea_nom','TSou_hea_nom',
                        'coeQ_coo','coeQ_hea','coeP_coo','coeP_hea', 'calc_method'}

        rejected_keys = set(kwargs.keys()) - set(allowed_keys)
        if rejected_keys:
            raise ValueError("Invalid arguments in constructor: {}".
                             format(rejected_keys))
        else:
            self.default_values()
            self.__dict__.update((k, v) for k, v in kwargs.items() 
            if k in allowed_keys)
            self.check_method()
            self.default_curve()
           
          
    def default_values(self):
        '''
        Reset the values to the default ones. 
        Default values for an air heat pump
        '''
        self.QLoa_hea_nom = 39040.0 
        self.QLoa_coo_nom = -39890 
        self.Phea_nom = 5130.0 
        self.Pcoo_nom = 4790.0 
        self.rev = True
        self.cpLoa = 1005.0 
        self.cpSou = 4180.0 
        self.min_plr = 0.1
        
        self.mLoa_flow_nom = 1.89
        self.mSou_flow_nom = 1.89
        self.TLoa_coo_nom = 10 + 273.15
        self.TSou_coo_nom = 10 + 273.15
        self.TLoa_hea_nom = 10 + 273.15
        self.TSou_hea_nom = 10 + 273.15
        
        self.coeQ_coo = pd.NA
        self.coeQ_hea = pd.NA
        self.coeP_coo = pd.NA
        self.coeP_hea = pd.NA
        
        self.calc_method = 'temp'

            
    def default_curve(self):
        '''
        Check if the boiler efficiency curve coefficients were
        defined manually. Otherwise define default curve coefficients.
        
        Default values 'temp_mass' -> https://github.com/lbl-srg/modelica-buildings/blob/master/Buildings/Fluid/HeatPumps/Data/EquationFitReversible/EnergyPlus.mo
        Default values 'temp' -> 
        
        Parameters:
    	----------	           
        § coeQ_hea: array of floats 
            coefficient heat pump heating heat flow efficiency curve       
        § coeQ_coo: array of floats 
            coefficient heat pump cooling heat flow efficiency curve       
        § coeP_hea: array of floats
            coefficient heat pump heating power efficiency curve       
        § coeP_coo: array of floats
            coefficient heat pump cooling power efficiency curve       
        '''
        if self.a+self.b+self.c+self.d == 0: 
            if self.eff == 'temp_mass':
                self.coeQ_hea = [-3.33491153,-0.51451946,4.51592706,0.01797107,0.155797661]
                self.coeP_hea = [-8.93121751,8.57035762,1.29660976,-0.21629222,0.033862378]
                self.coeQ_coo = [-1.52030596,3.46625667,-1.32267797,0.09395678,0.038975504]
                self.coeP_coo = [-8.59564386,0.96265085,8.69489229,0.02501669,-0.20132665]
            elif self.eff == 'temp':
                self.coeQ_hea = [-3.33491153,-0.51451946,4.51592706,0.01797107,0.155797661,0.02501669,]
                self.coeP_hea = [-8.93121751,8.57035762,1.29660976,-0.21629222,0.033862378,0.02501669,]
                self.coeQ_coo = [-1.52030596,3.46625667,-1.32267797,0.09395678,0.038975504,0.02501669,]
                self.coeP_coo = [-8.59564386,0.96265085,8.69489229,0.02501669,-0.20132665,0.02501669]
            else: 
                self.coeQ_hea = np.zeros(5)
                self.coeP_hea = np.zeros(5)
                self.coeQ_coo = np.zeros(5)
                self.coeP_coo = np.zeros(5)
        else:
            pass              

      
    def check_method(self):
        '''
        Check of the selected calculation method is defined in the model.
        Otherwise raise error.       
        Allowed methods:
        § temp: polynomical cubic approximation using the part load ratio
        § temp_mass: sigmoid function using the return water temperature
        § const: constant efficiency, assuming the nominal value        
        '''
        allowed_method = {'temp','temp_mass','const'}
        
        if self.calc_method not in allowed_method:
            raise ValueError("Invalid heat pump calculation method: {}".
                             format(self.eff))
        else:
            pass
        
        
    def hp_calc_temp(self, TLoa_out_set, TLoa_in, TSou_in, mLoa_flow):
        '''
        Calculates the heat pump thermal and electrical power as a function of 
        the source and sink inlet temperature. 
        Parameters:
    	----------	
        § TLoa_out_set: float
            Temperature set point load side, in Kelvin
        § TLoa_in: float
            Inlet temperature load side, in Kelvin
        § TSou_in: float
            Inlet temperature source side, in Kelvin
        § mLoa_flow: float
            Inlet mass flow load side, in kg/s
        Returns:
        -------
        § QLoa: float
            Heat pump thermal power, in W
        § P: float
            Heat pump electrical power, in W
        § TLoa_out: float
            Outlet temperature load side, in Kelvin
        '''      
        # Calcualte the required heat flux to reach the desired temperature set point
        QLoa_req = mLoa_flow*self.cpLoa*(TLoa_out_set-TLoa_in)
        
        inp_array = np.array([1,
                      TLoa_in,
                      TSou_in,
                      TLoa_in**2,
                      TSou_in**2,
                      TLoa_in*TSou_in])
                                  
        if (QLoa_req > 0) and (self.mod == 'heating'):
            # Heat pump is in heating mode                            
            QLoa_hea_ava = np.multiply(inp_array, np.array(self.coeQ_hea))*self.QLoa_hea_nom
            Phea_ava = np.multiply(inp_array, np.array(self.coeP_hea))*self.Phea_nom
        
            QLoa = min(QLoa_req, QLoa_hea_ava)
            plr = QLoa/QLoa_hea_ava
            if plr < self.min_plr:
                plr = self.min_plr
                QLoa = QLoa_hea_ava*self.min_plr
            P = Phea_ava*plr

        elif (QLoa_req < 0) and (self.mod == 'cooling'):
            # Heat pump is in cooling mode                            
            if self.rev == True:
                QLoa_coo_ava = np.multiply(inp_array, np.array(self.coeQ_coo))
                Pcoo_ava = np.multiply(inp_array, np.array(self.coeP_coo))   
                
                QLoa = max(QLoa_req, QLoa_coo_ava)
                plr = QLoa/QLoa_coo_ava
                if plr < self.min_plr:
                    plr = self.min_plr
                    QLoa = QLoa_coo_ava*self.min_plr
                P = Pcoo_ava*plr
            else:
                QLoa = 0
                P = 0
            
        else: 
            QLoa = 0
            P = 0
        
        TLoa_out = QLoa/(mLoa_flow*self.cpLoa) + TLoa_in
        return [QLoa, P, TLoa_out]


    def hp_calc_temp_mass(self, TLoa_out_set, TLoa_in, TSou_in, mLoa_flow, mSou_flow):
        '''
        Calculates the heat pump thermal and electrical power as a function of 
        the source and sink inlet temperature and mass flow. 
        Parameters:
    	----------	
        § TLoa_out_set: float
            Temperature set point load side, in Kelvin
        § TLoa_in: float
            Inlet temperature load side, in Kelvin
        § TSou_in: float
            Inlet temperature source side, in Kelvin
        § mLoa_flow: float
            Inlet mass flow load side, in kg/s
        § mSou_flow: float
            Inlet mass flow source side, in kg/s
        Returns:
        -------
        § QLoa: float
            Heat pump thermal power, in W
        § P: float
            Heat pump electrical power, in W
        § TLoa_out: float
            Outlet temperature load side, in Kelvin
        '''      
        # Calcualte the required heat flux to reach the desired temperature set point
        QLoa_req = mLoa_flow*self.cpLoa*(TLoa_out_set-TLoa_in)
        
        inp_array = np.array([1,
                      TLoa_in/self.TLoa_hea_nom,
                      TSou_in/self.TSou_hea_nom,
                      mLoa_flow/self.mLoa_flow_nom,
                      mSou_flow/self.mSou_flow_nom])
                                  
        if (QLoa_req > 0) and (self.mod == 'heating'):
            # Heat pump is in heating mode
            QLoa_hea_ava = np.multiply(inp_array, np.array(self.coeQ_hea))*self.QLoa_hea_nom
            Phea_ava = np.multiply(inp_array, np.array(self.coeP_hea))*self.Phea_nom
        
            QLoa = min(QLoa_req, QLoa_hea_ava)
            plr = QLoa/QLoa_hea_ava
            if plr < self.min_plr:
                plr = self.min_plr
                QLoa = QLoa_hea_ava*self.min_plr
            P = Phea_ava*plr

        elif (QLoa_req < 0) and (self.mod == 'cooling'):
            # Heat pump is in cooling mode                            
            if self.rev == True:
                QLoa_coo_ava = np.multiply(inp_array, np.array(self.coeQ_coo))
                Pcoo_ava = np.multiply(inp_array, np.array(self.coeP_coo))   
                
                QLoa = max(QLoa_req, QLoa_coo_ava)
                plr = QLoa/QLoa_coo_ava
                if plr < self.min_plr:
                    plr = self.min_plr
                    QLoa = QLoa_coo_ava*self.min_plr
                P = Pcoo_ava*plr
            else:
                QLoa = 0
                P = 0
            
        else: 
            QLoa = 0
            P = 0
        
        TLoa_out = QLoa/(mLoa_flow*self.cpLoa) + TLoa_in
        return [QLoa, P, TLoa_out]

        
        
    def output(self, TLoa_out_set, TLoa_in, TSou_in, mLoa_flow, mSou_flow, 
               ctrl=1, mod = 'heating', pr=True):
        '''
        Calculates the heat pump thermal and electrical power as a function of 
        the source and sink inlet temperature and mass flow. Gives the output 
        according to additional parameters of the heat pump
        Parameters:
    	----------	
        § TLoa_out_set: float
            Temperature set point load side, in Kelvin
        § TLoa_in: float
            Inlet temperature load side, in Kelvin
        § TSou_in: float
            Inlet temperature source side, in Kelvin
        § mLoa_flow: float
            Inlet mass flow load side, in kg/s
        § mSou_flow: float
            Inlet mass flow source side, in kg/s
        § ctrl: boolean
            Heat pump control flag
        § mod: string
            Heat pump mode ('heating' or 'cooling')            
        Returns:
        -------
        § QLoa: float
            Heat pump thermal power, in W
        § P: float
            Heat pump electrical power, in W
        § TLoa_out: float
            Outlet temperature load side, in Kelvin
        '''   
        self.TLoa_in = TLoa_in - 273.15 # inlet fluid temperature, calculation in °C
        self.TSou_in = TSou_in - 273.15 # inlet fluid temperature, calculation in °C
        self.mLoa_flow = mLoa_flow # inlet fluid flow rate
        self.mSou_flow = mSou_flow # inlet fluid flow rate
        self.ctrl = ctrl # control signal
        self.mod = mod # heat pump mode - heating or cooling
        self.TLoa_out_set = TLoa_out_set - 273.15 # requested flow temperature, calculation in °C
    
        if (self.TLoa_out_set - self.mLoa_in <= 0.001) or (self.ctrl <= 0):
            # Heat pump is OFF
            self.QLoa = 0.0
            self.P = self.T_in + 273.15
            self.TLoa_out = self.TLoa_in
            self.COP = 0.0
            self.QSou = 0.0

        else:
            # Heat pump is ON
            # Calculation of heat and power (load)
            if self.calc_method == 'temp_mass':
                [self.QLoa, self.P, self.TLoa_out] = self.hp_calc_temp_mass(TLoa_out_set, 
                                                                            TLoa_in, 
                                                                            TSou_in, 
                                                                            self.mLoa_flow, 
                                                                            self.mSou_flow)
            elif self.calc_method == 'temp':
                [self.QLoa, self.P, self.TLoa_out] = self.hp_calc_temp(TLoa_out_set, 
                                                                       TLoa_in, 
                                                                       TSou_in, 
                                                                       self.mLoa_flow)           
            else:
                [self.QLoa, self.P, self.TLoa_out] = [0,0,self.TLoa_in]
        
        # Further variables
        self.QSou = self.QLoa - self.P
        self.TSou_out = self.QSou/(self.mSou_in*self.cpSou) + TSou_in
        
        if pr:
            return [self.QLoa, self.P, self.TLoa_out]



# class Chiller_VRF: #TO DO
#     """ TO DO
#     Model of a heat pump using a characteristic curve fit, for space heating/cooling and DHW.
    
#     PLR calculation method -> doi.org/10.1007/s11708-018-0596-5
#     temp_mass calculation method -> https://hvac.okstate.edu/sites/default/files/pubs/theses/MS/27-Tang_Thesis_05.pdf
#     PLR + T_in calculation method -> doi.org/10.1016/j.enconman.2017.01.016
#     """
#     def __init__(self, **kwargs): 
#         '''TO DO
#         Initial conditions for the model
#         Allowed keyword arguments (**kwargs):
#     	----------	
#         § Qflow_nom: float
#             nominal thermal power of the boiler in W
#         § eta_nom: float
#             nominal thermal efficiency of the gas combustion process, in fraction
#         § cp: float
#             specific heat capacity secondary circuit, kJ/kgK
#         § min_plr: float
#             mininum partload fraction of calculated heat pump, in fraction
#         § eff: string
#             calculation method of the efficiency curve              
#         '''
                                         
#         allowed_keys = {'QLoa_hea_nom', 'QLoa_coo_nom','Pcoo_nom', 'Phea_nom', 'rev',
#                         'cpLoa', 'cpSou', 'min_plr','mLoa_flow_nom','mSou_flow_nom',
#                         'TLoa_coo_nom','TSou_coo_nom','TLoa_hea_nom','TSou_hea_nom',
#                         'coeQ_coo','coeQ_hea','coeP_coo','coeP_hea', 'calc_method'}

#         rejected_keys = set(kwargs.keys()) - set(allowed_keys)
#         if rejected_keys:
#             raise ValueError("Invalid arguments in constructor: {}".
#                              format(rejected_keys))
#         else:
#             self.default_values()
#             self.__dict__.update((k, v) for k, v in kwargs.items() 
#             if k in allowed_keys)
#             self.check_method()
#             self.default_curve()
           
          
#     def default_values(self):
#         '''TO DO
#         Reset the values to the default ones. 
#         EnergyPlus default values for air heat pump
#         '''
#         self.QLoa_hea_nom = 39040.0 
#         self.QLoa_coo_nom = -39890 
#         self.Phea_nom = 5130.0 
#         self.Pcoo_nom = 4790.0 
#         self.rev = True
#         self.cpLoa = 1005.0 
#         self.cpSou = 4180.0 
#         self.min_plr = 0.1
        
#         self.mLoa_flow_nom = 1.89
#         self.mSou_flow_nom = 1.89
#         self.TLoa_coo_nom = 10 + 273.15
#         self.TSou_coo_nom = 10 + 273.15
#         self.TLoa_hea_nom = 10 + 273.15
#         self.TSou_hea_nom = 10 + 273.15
        
#         self.coeQ_coo = pd.NA
#         self.coeQ_hea = pd.NA
#         self.coeP_coo = pd.NA
#         self.coeP_hea = pd.NA
        
#         self.calc_method = 'temp'

            
#     def default_curve(self):
#         '''TO DO
#         Check if the boiler efficiency curve coefficients were
#         defined manually. Otherwise define default curve coefficients.
        
#         Default values 'temp_mass' -> https://github.com/lbl-srg/modelica-buildings/blob/master/Buildings/Fluid/HeatPumps/Data/EquationFitReversible/EnergyPlus.mo
#         Default values 'temp' -> 
#         Default values 'vrf' -> https://www.osti.gov/servlets/purl/1079215
        
#         Parameters:
#     	----------	           
#         § a: float 
#             coefficient "a" for boiler efficiency curve       
#         § b: float 
#             coefficient "b" for boiler efficiency curve       
#         § c: float
#             coefficient "c" for boiler efficiency curve       
#         § d: float
#             coefficient "d" for boiler efficiency curve 

               
#         '''
#         if self.a+self.b+self.c+self.d == 0: 
#             if self.eff == 'vrf':
#                 self.coeQ_hea = [-3.33491153,-0.51451946,4.51592706,0.01797107,0.155797661,0.02501669,]
#                 self.coeP_hea = [-8.93121751,8.57035762,1.29660976,-0.21629222,0.033862378,0.02501669,]
#                 self.coeQ_coo = [-1.52030596,3.46625667,-1.32267797,0.09395678,0.038975504,0.02501669,]
#                 self.coeP_coo = [-8.59564386,0.96265085,8.69489229,0.02501669,-0.20132665,0.02501669]
#             else: 
#                 self.coeQ_hea = np.zeros(5)
#                 self.coeP_hea = np.zeros(5)
#                 self.coeQ_coo = np.zeros(5)
#                 self.coeP_coo = np.zeros(5)
#         else:
#             pass              

      
#     def check_method(self):
#         '''TO DO
#         Check of the selected calculation method is defined in the model.
#         Otherwise raise error.       
#         Allowed methods:
#         § temp: polynomical cubic approximation using the part load ratio
#         § temp_mass: sigmoid function using the return water temperature
#         § vrf: function based on the part load ratio nad return water temperature
#         § const: constant efficiency, assuming the nominal value        
#         '''
#         allowed_method = {'temp','temp_mass','vrf','const'}
        
#         if self.calc_method not in allowed_method:
#             raise ValueError("Invalid heat pump calculation method: {}".
#                              format(self.eff))
#         else:
#             pass
        
        
#     def HP_calc_temp(self, TLoa_out_set, TLoa_in, TSou_in, mLoa_flow):
#         '''TO DO
#         Calculates the boiler efficiency - eta
#         Cubic efficiency curve
#         Parameters:
#     	----------	
#         § plr: float
#             part load ratio, in fraction
#         Returns:
#         -------
#         § eta: float
#             efficiency of the combustion process
#         '''      
#         QLoa_req = mLoa_flow*self.cpLoa*(TLoa_out_set-TLoa_in)
        
#         inp_array = np.array([1,
#                       TLoa_in,
#                       TSou_in,
#                       TLoa_in**2,
#                       TSou_in**2,
#                       TLoa_in*TSou_in])
                                  
#         if (QLoa_req > 0) and (self.mod == 'heating'):
#             # Heat pump is in heating mode                            
#             QLoa_hea_ava = np.multiply(inp_array, np.array(self.coeQ_hea))*self.QLoa_hea_nom
#             Phea_ava = np.multiply(inp_array, np.array(self.coeP_hea))*self.Phea_nom
        
#             QLoa = min(QLoa_req, QLoa_hea_ava)
#             plr = QLoa/QLoa_hea_ava
#             if plr < self.min_plr:
#                 plr = self.min_plr
#                 QLoa = QLoa_hea_ava*self.min_plr
#             P = Phea_ava*plr

#         elif (QLoa_req < 0) and (self.mod == 'cooling'):
#             # Heat pump is in cooling mode                            
#             if self.rev == True:
#                 QLoa_coo_ava = np.multiply(inp_array, np.array(self.coeQ_coo))
#                 Pcoo_ava = np.multiply(inp_array, np.array(self.coeP_coo))   
                
#                 QLoa = max(QLoa_req, QLoa_coo_ava)
#                 plr = QLoa/QLoa_coo_ava
#                 if plr < self.min_plr:
#                     plr = self.min_plr
#                     QLoa = QLoa_coo_ava*self.min_plr
#                 P = Pcoo_ava*plr
#             else:
#                 QLoa = 0
#                 P = 0
            
#         else: 
#             QLoa = 0
#             P = 0
        
#         TLoa_out = QLoa/(mLoa_flow*self.cpLoa) + TLoa_in
#         return [QLoa, P, TLoa_out]


        
        
#     def output(self, TLoa_out_set, TLoa_in, TSou_in, mLoa_flow, mSou_flow, 
#                ctrl=1, mod = 'heating', pr=True):
#         '''TO DO
#         Calculates the boiler rated power as a function of different values
#         Sigmoid function of the efficiency
#         Parameters:
#     	----------	
#         § T_out_set: float
#             temperature set point of the supply water, in Kelvin
#         § T_in: float
#             return temperature of the fuel flow, in Kelvin
#         § M_in: float
#             water mass flow rate, in kg/s
#         § ctrl: integer
#             boiler control flag, 0 = off, 1 = on
#         § pr: boolean
#             print results
#         Returns:
#         -------
#         § Qflow_fuel: float
#             consumed gas mass flow, in kg/s
#         '''
#         self.TLoa_in = TLoa_in - 273.15 # inlet fluid temperature, calculation in °C
#         self.TSou_in = TSou_in - 273.15 # inlet fluid temperature, calculation in °C
#         self.mLoa_flow = mLoa_flow # inlet fluid flow rate
#         self.mSou_flow = mSou_flow # inlet fluid flow rate
#         self.ctrl = ctrl # control signal
#         self.mod = mod # heat pump mode - heating or cooling
#         self.TLoa_out_set = TLoa_out_set - 273.15 # requested flow temperature, calculation in °C

#         if (self.TLoa_out_set - self.mLoa_in <= 0.001) or (self.ctrl <= 0):
#             # Heat pump is OFF
#             self.QLoa = 0.0
#             self.P = self.T_in + 273.15
#             self.TLoa_out = self.TLoa_in
#             self.COP = 0.0
#             self.QSou = 0.0

#         else:
#             # Heat pump is ON
#             # Calculation of heat and power (load)
#             if self.calc_method == 'temp_mass':
#                 [self.QLoa, self.P, self.TLoa_out] = self.HP_calc_temp_mass(TLoa_out_set, 
#                                                                             TLoa_in, 
#                                                                             TSou_in, 
#                                                                             self.mLoa_flow, 
#                                                                             self.mSou_flow)
#             elif self.calc_method == 'temp':
#                 [self.QLoa, self.P, self.TLoa_out] = self.HP_calc_temp(TLoa_out_set, 
#                                                                        TLoa_in, 
#                                                                        TSou_in, 
#                                                                        self.mLoa_flow)           
#             elif self.calc_method == 'vrf':
#                 [self.QLoa, self.P, self.TLoa_out] = self.HP_calc_vrf(TLoa_out_set, 
#                                                                       TLoa_in, 
#                                                                       TSou_in, 
#                                                                       self.mLoa_flow, 
#                                                                       self.mSou_flow)
#             else:
#                 [self.QLoa, self.P, self.TLoa_out] = [0,0,self.TLoa_in]
        
#         # Further variables
#         self.QSou = self.QLoa - self.P
#         self.TSou_out = self.QSou/(self.mSou_in*self.cpSou) + TSou_in
        
#         if pr:
#             return [self.QLoa, self.P, self.TLoa_out]
            

class HeatRecovery: 
    """ 
    Model for air-to-air counterflow heat recovery systems without latent heat transfer.
    The heat transfer is calculated following the NTU-method (number of transfer units).
    This model can be used as well as an economizer in HVAC systems. 
    Built based on the following reference, page 11 (https://buildmedia.readthedocs.org/media/pdf/ht/latest/ht.pdf)
    """
    def __init__(self, **kwargs):
        """
        Parameters:
    	----------
        § cp_a: float
            specific heat capacitity fluid a, in J/kgK
        § cp_b: float
            specific heat capacitity fluid b, in J/kgK
        § UA: float
            overall heat transfer coefficient, in W/K
        """       
        allowed_keys = {'cp_a', 'cp_b','UA'}

        rejected_keys = set(kwargs.keys()) - set(allowed_keys)
        if rejected_keys:
            raise ValueError("Invalid arguments in constructor: {}".
                             format(rejected_keys))
        else:
            self.default_values()
            self.__dict__.update((k, v) for k, v in kwargs.items() 
            if k in allowed_keys)
          
          
    def default_values(self):
        '''
        Reset the values to the default ones. 
        '''
        self.cp_a = 4180.0
        self.cp_b = 4180.0
        self.UA = 1000.0
    
    
    def eff_NTU(self, mflow_a, mflow_b, pr=False):
        """
        Calculates the effectiveness of the heat transfer using the NTU-method, 
        as a function of the inlet mass flow of both fluids
        Parameters:
    	----------
        § cp_a: float
            specific heat capacitity fluid a, in J/kgK
        § cp_b: float
            specific heat capacitity fluid b, in J/kgK
        § UA: float
            overall heat transfer coefficient, in W/K
        § pr: boolean
            print results
        """   
    
        #Minimum and maximum capacity flow rate, for NTU calculation
        self.C_a    = self.cp_a * mflow_a
        self.C_b    = self.cp_b * mflow_b
        self.C_Max  = max(self.C_a,self.C_b)
        self.C_Min  = min(self.C_a,self.C_b)

        #Effectiveness calculation
        self.Cr = self.C_Min/self.C_Max
        self.NTU = self.UA/self.C_Min
        self.check_NTU_min = abs(1.0 - self.R)

        if self.check_NTU_min < 0.01:
            self.eff = self.NTU/(self.NTU+1.0)
        else:
            # calculation for counter flow
            self.eff = (1.0 - np.exp(-self.NTU*(1.0-self.Cr))) / (1.0-self.Cr*np.exp(-self.NTU*(1.0-self.Cr)))  
        
        if pr:
            return [self.eff, self.C_Min]
        

    def output(self, T_a_in, mflow_a, T_b_in, mflow_b, pr=False):
        """ 
        Calculates the heat exchanged between the fluids as a 
        function of the inlet temperatures and mass flow rates
        Parameters:
    	----------	
        § T_a_in: float
            inlet temperature of fluid a, in Kelvin
        § T_b_in: float
            inlet temperature of fluid b, in Kelvin
        § mflow_a: float
            mass flow rate of fluid a, in kg/s
        § mflow_b: float
            mass flow rate of fluid b, in kg/s
        § pr: boolean
            print results
        Returns:
        -------
        § Qflow: float
            heat exchanged between fluids, in J
        § T_a_out: float
            outlet temperature of fluid a, in Kelvin
        § T_b_out: float
            outlet temperature of fluid b, in Kelvin
        """
        self.T_a_in = T_a_in
        self.mflow_a = mflow_a
        self.T_b_in = T_b_in
        self.mflow_b = mflow_b

        if self.mflow_a == 0.0 or self.mflow_b == 0.0:
            self.C_a            = 0.0
            self.C_b            = 0.0
            self.C_Max          = 0.0
            self.C_Min          = 0.0
            self.Cr              = 0.0
            self.NTU            = 0.0
            self.check_NTU_min  = 0.0
            self.eff            = 0.0
            self.Qflow          = 0.0
            self.T_a_out = self.T_a_in
            self.T_b_out = self.T_b_in
        else:
            #Effectiveness calculation
            self.eff_NTU(mflow_a, mflow_b)
            #Heat flow rate calculation
            self.Qflow  = self.eff*self.C_Min*(self.T_a_in-self.T_b_in)

            #Outlet temperatures calculation
            self.T_a_out = self.T_a_in - (self.eff*(self.C_Min/self.C_a)*(self.T_a_in-self.T_b_in))
            self.T_b_out = self.T_b_in + (self.eff*(self.C_Min/self.C_b)*(self.T_a_in-self.T_b_in)) 
        
        if pr:
            return [self.Qflow, self.T_a_out, self.T_b_out]



if __name__ == '__main__':
    pass
