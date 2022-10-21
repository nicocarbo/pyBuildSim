# -*- coding: utf-8 -*-
"""
author: Nicolas Carbonare
mail: nicocarbonare@gmail
last updated: June 06, 2021
"""

import numpy as np
import pandas as pd

	
class WindowOpening_Andersen:
    '''  
    Logistic regression for residential window opening from 
    Andersen et al (2013). 
    http://dx.doi.org/10.1016/j.buildenv.2013.07.005
    '''
    def __init__(self,
                 init_state = 0,
                 room = 'living',
                 group = 4.0):
        '''
        Initial conditions for the model
        Parameters:
    	----------	
        § init_state: integer (default = 0)
            initial window state
        § room: string (default = 'living')
            type of room (only 'bedroom' or 'living')
        § group: integer (default = 4)
            group number according to the publication
        '''
        self.state = init_state
        self.room = room
        self.group = group
        
    def coef_open(self, room, group):
        '''
        Selects the coefficient matrix for the window opening action
        Parameters:
    	----------	
        § room: string (default = 'bedroom')
            type of room (only 'bedroom' or 'living')
        § group: integer (default = 4)
            group number according to the publication
        Returns:
        -------
        § coef_mat: matrix of floats
            matrix of coefficients
        '''        
        bedroom_2 = pd.DataFrame({
                'Intercept': [-13.49],
                'Tout': [0],
                'Solarrad': [0],
                'Solarhs': [0],
                'RHout': [0],
                'Tin': [0],
                'CO2': [1.40],
                'RHin': [0]
                })
        # bedroom_2.index = ('All')

        living_2 = pd.DataFrame({
                'Intercept': [-13.49],
                'Tout': [0],
                'Solarrad': [0],
                'Solarhs': [0],
                'RHout': [0],
                'Tin': [0],
                'CO2': [1.40],
                'RHin': [0]
                })
        # living_2.index = ('All')
        
        bedroom_3 = pd.DataFrame({
                'Intercept': [-17.69, -15.51, -17.09, -18.23, -17.13],
                'Tout': [0]*5,
                'Solarrad': [0]*5,
                'Solarhs': [0]*5,
                'RHout': [0]*5,
                'Tin': [0]*5,
                'CO2': [1.75]*5,
                'RHin': [0]*5
                })
        bedroom_3.index = ('Night', 'Morning', 'Day', 'Afternoon', 'Evening')

        living_3 = pd.DataFrame({
                'Intercept': [-17.69, -15.51, -17.09, -18.23, -17.13],
                'Tout': [0]*5,
                'Solarrad': [0]*5,
                'Solarhs': [0]*5,
                'RHout': [0]*5,
                'Tin': [0]*5,
                'CO2': [1.75]*5,
                'RHin': [0]*5
                })
        living_3.index = ('Night', 'Morning', 'Day', 'Afternoon', 'Evening')
        
        bedroom_4 = pd.DataFrame({
                'Intercept': [-18.53]*3,
                'Tout': [-0.019]*3,
                'Solarrad': [0.18]*3,
                'Solarhs': [0.057]*3,
                'RHout': [0.029]*3,
                'Tin': [0.10]*3,
                'CO2': [1.16]*3,
                'RHin': [0]*3
                })
        bedroom_4.index = ('Winter', 'Spring', 'Summer')

        living_4 = pd.DataFrame({
                'Intercept': [-18.53]*3,
                'Tout': [-0.019]*3,
                'Solarrad': [0.35]*3,
                'Solarhs': [0.057]*3,
                'RHout': [0.029]*3,
                'Tin': [-0.38]*3,
                'CO2': [0.30]*3,
                'RHin': [0]*3
                })
        living_4.index = ('Winter', 'Spring', 'Summer')
    
        return locals()['{:.8}_%g'.format(room) %group]
    

    def coef_close(self, room, group):
        '''
        Selects the coefficient matrix for the window closing action
        Parameters:
    	----------	
        § room: string (default = 'bedroom')
            type of room (only 'bedroom' or 'living')
        § group: integer (default = 4)
            group number according to the publication
        Returns:
        -------
        § coef_mat: matrix of floats
            matrix of coefficients
        '''             
        bedroom_2 = pd.DataFrame({
                'Intercept': [-4.75],
                'Tout': [0.03],
                'Solarrad': [0.59],
                'Solarhs': [-0.06],
                'RHout': [0],
                'Tin': [0],
                'CO2': [0],
                'RHin': [0]
                })
        #bedroom_2.index = ('All')

        living_2 = pd.DataFrame({
                'Intercept': [4.19],
                'Tout': [-0.26],
                'Solarrad': [0.04],
                'Solarhs': [-0.06],
                'RHout': [0],
                'Tin': [0],
                'CO2': [0],
                'RHin': [0]
                })
        #living_2.index = ('All')
                
        bedroom_3 = pd.DataFrame({
                'Intercept': [-2.68, -0.51, -7.67, -12.78,-13.22],
                'Tout': [0.01, 0.12, -0.13, -0.07, -0.09],
                'Solarrad': [0]*5,
                'Solarhs': [-0.08]*5,
                'RHout': [0]*5,
                'Tin': [0.4, 0.15, 0.21, 0.70, 0.60],
                'CO2': [0]*5,
                'RHin': [-0.25, -0.16, 0.06, -0.15, -0.07]
                })
        bedroom_3.index = ('Night', 'Morning', 'Day', 'Afternoon', 'Evening') 

        living_3 = pd.DataFrame({
                'Intercept': [14.68, 16.85, 9.69, 4.57, 4.13],
                'Tout': [-0.13, -0.01, -0.27, -0.20, -0.22],
                'Solarrad': [0]*5,
                'Solarhs': [-0.08]*5,
                'RHout': [0]*5,
                'Tin': [-0.25, -0.51, -0.45, 0.05, -0.05],
                'CO2': [0]*5,
                'RHin': [-0.25, -0.16, 0.06, -0.15, -0.07]
                })
        living_3.index = ('Night', 'Morning', 'Day', 'Afternoon', 'Evening') 
        
        bedroom_4 = pd.DataFrame({
                'Intercept': [-4.28, -2.98, -4.94],
                'Tout': [-0.038, -0.147, -0.057],
                'Solarrad': [0.13]*3,
                'Solarhs': [-0.089]*3,
                'RHout': [-0.028]*3,
                'Tin': [0]*3,
                'CO2': [0]*3,
                'RHin': [0.063]*3
                })
        bedroom_4.index = ('Winter', 'Spring', 'Summer')
        
        living_4 = pd.DataFrame({
                'Intercept': [-0.62, 0.68, -1.28],
                'Tout': [-0.17, -0.27, -0.18],
                'Solarrad': [0.13]*3,
                'Solarhs': [-0.089]*3,
                'RHout': [-0.028]*3,
                'Tin': [0]*3,
                'CO2': [0]*3,
                'RHin': [0.036]*3
                })
        living_4.index = ('Winter', 'Spring', 'Summer')
    
        return locals()['{:.8}_%g'.format(room) %group]
    

    def vartransform(self, CO2, Solarrad):
        '''
        Transform the variables required by the model
        Parameters:
        ----------
        $ CO2: float 
            CO2 value in ppm
        § Solarrad: float  
            Solar radation in W/m2
        Returns:
    	-------
        § array with log-transformations 
        '''    
        return [np.log(CO2), 
                np.log(Solarrad + 1)]
    
    
    def set_coef(self, state):
        
        if state == 0:
            return self.coef_open(self.room, self.group)
        else:
            return self.coef_close(self.room, self.group)
        

    def predict_proba(self, Tout, Solarrad, Solarhs,
                      RHout, Tin, CO2, RHin, time ='All'):
        '''  
        Predict the probability of an action taking place        
        Parameters:
        ----------
        § Tout: float 
			Outdoor temperature in °C
        § Solarrad: float  
            Solar radation in W/m2
        § Solarhs: float  
            Solar hours in h
		§ RHout: float 
			Outdoor relative humidity in % (i.e. 50, not 0.5)
		§ Tin: float 
			Indoor temperature in °C
        $ CO2: float 
            CO2 value in ppm
        § RHin: float 
			Indoor relative humidity in % (i.e. 50, not 0.5)
        § time: string (default = 'All')
			Time of the day of the year:
                - ('Winter', 'Spring', 'Summer')
                - ('Night', 'Morning', 'Day', 'Afternoon', 'Evening') 
        Returns:
		-------
        § p: float 
			action probability
        '''

        [CO2_t, Solarrad_t] = self.vartransform(CO2, Solarrad)
        
        self.coef_mat = self.set_coef(self.state)
        self.coef_arr = self.coef_mat.loc[time]
        
        self.p  = 1/(np.exp((-1)*(    
                     self.coef_arr[0] + Tout*self.coef_arr[1] + \
                     Solarrad_t*self.coef_arr[2] + Solarhs*self.coef_arr[3] + \
                     RHout*self.coef_arr[4] + Tin*self.coef_arr[5] + \
                     CO2_t*self.coef_arr[6] + RHin*self.coef_arr[7])) + 1)
                        
        return self.p        
        
    
    def update_win_state(self, Tout, Solarrad, Solarhs,
                         RHout, Tin, CO2, RHin, time = 'All', occ=1):
        '''  
        Update the state of the window by comparing the action 
        probability with a random number from 0 to 1.         
        Parameters:
        ----------
        § Tout: float 
			Outdoor temperature in °C
        § Solarrad: float  
            Solar radation in W/m2
        § Solarhs: float  
            Solar hours in h
		§ RHout: float 
			Outdoor relative humidity in % (i.e. 50, not 0.5)
		§ Tin: float 
			Indoor temperature in °C
        $ CO2: float 
            CO2 value in ppm
        § RHin: float 
			Indoor relative humidity in % (i.e. 50, not 0.5)
        § time: string (default = 'All')
			Time of the day of the year:
                - ('Winter', 'Spring', 'Summer')
                - ('Night', 'Morning', 'Day', 'Afternoon', 'Evening') 
        § occ: integer (default = 1) 
            Occupancy status (1 = Present, 0 = Absent)
        '''
        
        self.predict_proba(Tout, Solarrad, Solarhs,
                           RHout, Tin, CO2, RHin, time)
        
        if self.state == 0:
            self.state = np.greater(self.p, np.random.rand())*occ
        else:
            self.state = (1-np.greater(self.p, np.random.rand())*occ)*self.state
               

class WindowOpening_Cali:
    '''  
    Logistic regression for residential window opening from 
    Calì et al (2016). 
    http://dx.doi.org/10.1016/j.buildenv.2016.03.024
    '''
    def __init__(self,
                 init_state = 0,
                 room = 'living'):
        '''
        Initial conditions for the model
        Parameters:
    	----------	
        § init_state: integer (default = 0)
            initial window state
        § room: string (default = 'living')
            type of room (only 'bedroom' or 'living')
        '''
        self.state = init_state
        self.room = room


    def intercept_open(self, hour):
        '''
        Selects the intercept of the regression given 
        the hour of the opening action. There are three
        possible outcomes:
            § Night: between 23 and 6 am
            § Morning: between 7 and 10 am
            § Rest of the day: every other hour
            
        Parameters:
    	----------	
        § hour: integer
            hour value (i.e. if 18:45, hour = 18)
        Returns:
        -------
        § intercept_close_array: float
            logistic regression opening intercept
        '''        
        
        intercept_open_array = [-10.089, -8.214, -7.795]
        
        if hour < 6: 
            return intercept_open_array[0]
        elif hour > 22:
            return intercept_open_array[0]
        elif ((hour > 6)and(hour<10)):
            return intercept_open_array[1]
        else:
            return intercept_open_array[2]
            
            
    def intercept_close(self, hour):
        '''
        Selects the intercept of the regression given 
        the hour of the closing action. There are three
        possible outcomes:
            § Night: between 23 and 6 am
            § Morning: between 7 and 10 am
            § Rest of the day: every other hour
            
        Parameters:
    	----------	
        § hour: integer
            hour value (i.e. if 18:45, hour = 18)
        Returns:
        -------
        § intercept_close_array: float
            logistic regression closing intercept
        '''        
        
        intercept_close_array = [2.539, 3.317, 3.955]
                
        if hour < 6: 
            return intercept_close_array[0]
        elif hour > 22:
            return intercept_close_array[0]
        elif ((hour > 6)and(hour<10)):
            return intercept_close_array[1]
        else:
            return intercept_close_array[2]
        
        
    def opening(self):
        '''
        Selects the coefficient matrix for the window opening action

        Returns:
        -------
        § coef_mat: matrix of floats
            matrix of coefficients
        '''        
        opening_coef = np.array([0.134,-551.15,0,0,0])

        return opening_coef
    

    def closing(self):
        '''
        Selects the coefficient matrix for the window closing action

        Returns:
        -------
        § coef_mat: matrix of floats
            matrix of coefficients
        '''        
        closing_coef = np.array([-0.268,-785.7,-0.058,-0.089,0.027])

        return closing_coef
    

    def predict_proba(self, Tin, CO2, RHin, DAT, RHout, hour=12):
        '''  
        Predict the probability of an action taking place        
        Parameters:
        ----------
		§ Tin: float 
			Indoor temperature in °C
        $ CO2: float 
            CO2 value in ppm
        § RHin: float 
			Indoor relative humidity in % (i.e. 50, not 0.5)
        § DAT: float 
			Daily average outdoor temperature in °C
		§ RHout: float 
			Outdoor relative humidity in % (i.e. 50, not 0.5)
        § hour: integer (default = 'All')
			Hour of the day
        Returns:
		-------
        § p: float 
			action probability
        '''
        
        if self.state == 0:
            self.coef_arr = self.opening()
            self.intercept = self.intercept_open(hour)       
        else:
            self.coef_arr = self.closing()
            self.intercept = self.intercept_close(hour)       
            
        self.p  = 1/(np.exp((-1)*(    
                     self.intercept + Tin*self.coef_arr[0] + \
                     (1/CO2)*self.coef_arr[1] + RHin*self.coef_arr[2] + \
                     DAT*self.coef_arr[3] + RHout*self.coef_arr[4])) + 1)
                        
        return self.p        
        
    
    def update_win_state(self, Tin, CO2, RHin, DAT, RHout, hour=12, occ=1):
        '''  
        Update the state of the window by comparing the action 
        probability with a random number from 0 to 1.         
        Parameters:
        ----------
		§ Tin: float 
			Indoor temperature in °C
        $ CO2: float 
            CO2 value in ppm
        § RHin: float 
			Indoor relative humidity in % (i.e. 50, not 0.5)
        § DAT: float 
			Daily average outdoor temperature in °C
		§ RHout: float 
			Outdoor relative humidity in % (i.e. 50, not 0.5)
        § hour: integer (default = 'All')
			Hour of the day
        '''
        
        self.predict_proba(Tin, CO2, RHin, DAT, RHout, hour)
        
        if self.state == 0:
            self.state = np.greater(self.p, np.random.rand())*occ
        else:
            self.state = (1-np.greater(self.p, np.random.rand())*occ)*self.state



class WindowOpening_Schweiker:
    '''  
    Logistic regression for residential window opening from 
    Schweiker et al (2012).
      
    Parameters:
	----------	
        § A = 0.711 (intercept)
        § B = 0.3813 (coefficient T_out)
        § C = -0.3077 (coefficient T_in)

    '''
    def __init__(self,
                 A              = 0.711,
                 B              = 0.3813,
                 C              = -0.3077,        
                 state          = 0
                 ):
        self.A                  = A
        self.B                  = B
        self.C                  = C
        self.state              = state


    def predict_proba(self, Tin, Tout):
        '''  
        Predict the probability of having an open window        
        Parameters:
        ----------
		§ Tin: float 
				Indoor temperature in °C
        § Tout: float 
				Outdoor temperature in °C
        Returns:
		-------
        § p: float 
			action probability
        '''
        
        self.p = np.exp(self.A + Tout*self.B + Tin*self.C)/(
                        np.exp(self.A + Tout*self.B + Tin*self.C) + 1)
                        
        return self.p 
    
    
    def update_win_state(self, Tin, Tout, occ=1):
        '''  
        Calculate the prediciton of window state 
		
        Parameters:
        ----------
		§ Tin: float 
				Indoor temperature in °C
        § Tout: float 
				Outdoor temperature in °C
        § occ: integer  
				Occupancy status (1 = Present, 0 = Absent)
        '''
        self.predict_proba(Tin, Tout)
                
        self.state = np.greater(self.p, np.random.rand())*occ



class Temp_SP_DIN15251:
    '''
    Controller to set the temperature set point in a room
    The set point is defined given a certain time (night set back).
    The values are obtained from the DIN EN 15251 - 2007.       	
    '''
    def __init__(self, 
                 time_sb        = [7, 22],
                 temp_sb        = [16, 20]
                 ):
        '''
		Parameters:
		----------	
        § time_sb: 1d array of integers (default = [7, 22])
				array-like variable for turning on and off
                the night set back
        § temp_sb: 1d array of floats (default = [16, 20])
				array-like variable for the desired temperatures
		'''
        self.time_sb            = time_sb
        self.temp_sb            = temp_sb

    def calc(self, hour): 
        '''  
        Calculate the set back 
        Parameters:
		----------
        § hour: integer
				Time of the day
		$ rr: boolean (default = False)
				Show the output of the function
        Returns:
		-------
        § output: float 
				Temperature set point for heating
        '''

        self.output = self.temp_sb[0] + (self.temp_sb[1] - self.temp_sb[0])*(
                           np.greater(hour, self.time_sb[0]) * 
                           np.greater(self.time_sb[1], hour)) 

        return self.output 
      


class Temp_SP_Fabi:
    '''  
    Logistic regression for residential heating set point determination
    Probabilistic model obtained from Fabi et al (2013). 
    '''
    def __init__(self, **kwargs):
        '''
        Initial conditions for the model
        Allowed keyword arguments (**kwargs):
		------------------------------------
        § init_SP: integer (default = 21)
            initial temperature set point in degrees Celsius
        § user_type: string (default = 'medium')
            type of user (onl 'medium', 'passive' or 'active')
        '''
        allowed_keys = {'init_SP', 'user_type'}
        
        self.default_values()
        self.__dict__.update((k, v) for k, v in kwargs.items() 
        if k in allowed_keys)
        
        rejected_keys = set(kwargs.keys()) - set(allowed_keys)
        if rejected_keys:
            raise ValueError("Invalid arguments in constructor: {}".
                             format(rejected_keys))

        self.SP = self.init_SP
        
    def default_values(self):
        '''
        Default conditions for the model
        Parameters:
    	----------	
        § init_SP: integer (default = 21)
            initial temperature set point in degrees Celsius
        § user_type: string (default = 'medium')
            type of user (onl 'medium', 'passive' or 'active')
        '''
        self.init_SP = 21
        self.max_SP = 27
        self.min_SP = 17
        self.user_type = 'medium'

        
    def coef_up(self, user_type):
        '''
        Selects the coefficient matrix for the window opening action
        Parameters:
    	----------	
        § room: string (default = 'bedroom')
            type of room (only 'bedroom' or 'living')
        § group: integer (default = 4)
            group number according to the publication
        Returns:
        -------
        § coef_mat: matrix of floats
            matrix of coefficients
        '''
        active = pd.DataFrame({
                'Intercept': [-4.286, -0.6264, -0.839, -0.8663, -2.1435],
                'RHin': [-0.085]*5,
                'Tout': [-0.1441]*5,
                'Solarrad': [0]*5,
                'Windsp': [0]*5
                })
        active.index = ('Night', 'Morning', 'Day', 'Afternoon', 'Evening')
        
        
        medium = pd.DataFrame({
                'Intercept': [-7.6356]*5,
                'RHin': [0]*5,
                'Tout': [-0.2284]*5,
                'Solarrad': [0]*5,
                'Windsp': [0.3699]*5
                })
        medium.index = ('Night', 'Morning', 'Day', 'Afternoon', 'Evening')

        passive = pd.DataFrame({
                'Intercept': [-9.716]*5,
                'RHin': [0]*5,
                'Tout': [0]*5,
                'Solarrad': [0]*5,
                'Windsp': [0]*5
                })
        passive.index = ('Night', 'Morning', 'Day', 'Afternoon', 'Evening')
    
        return locals()['{:.8}'.format(user_type)]
    

    def coef_down(self, user_type):
        '''
        Selects the coefficient matrix for the window closing action
        Parameters:
    	----------	
        § room: string (default = 'bedroom')
            type of room (only 'bedroom' or 'living')
        § group: integer (default = 4)
            group number according to the publication
        Returns:
        -------
        § coef_mat: matrix of floats
            matrix of coefficients
        '''             
        active = pd.DataFrame({
                'Intercept': [-3.514]*5,
                'RHin': [0]*5,
                'Tout': [0]*5,
                'Solarrad': [-0.0194]*5,
                'Windsp': [0]*5
                })
        active.index = ('Night', 'Morning', 'Day', 'Afternoon', 'Evening')

        medium = pd.DataFrame({
                'Intercept': [-22.8446, -5.1599, -6.0973, -6.5805, -6.6572],
                'RHin': [0]*5,
                'Tout': [0]*5,
                'Solarrad': [0]*5,
                'Windsp': [0]*5
                })
        medium.index = ('Night', 'Morning', 'Day', 'Afternoon', 'Evening')

        passive = pd.DataFrame({
                'Intercept': [-14.2779]*5,
                'RHin': [0]*5,
                'Tout': [0]*5,
                'Solarrad': [0]*5,
                'Windsp': [1.0077]*5
                })
        passive.index = ('Night', 'Morning', 'Day', 'Afternoon', 'Evening')
    
        return locals()['{:.8}'.format(user_type)]
    
    
    
    def set_coef(self, user_type):
        
        self.coef_mat_up = self.coef_up(user_type)
        self.coef_mat_down = self.coef_down(user_type)
              

    def predict_proba(self, RHin, Tout, Solarrad, Windsp, time ='Day'):
        '''  
        Predict the probability of an action taking place        
        Parameters:
        ----------
        § RHin: float 
			Indoor relative humidity in % (i.e. 50, not 0.5)
        § Tout: float 
			Outdoor temperature in °C
        § Solarrad: float  
            Solar hours in h
		§ Windsp: float 
			Wind speed in m/s
        § time: string (default = 'Day')
			Time of the day:
                - ('Night', 'Morning', 'Day', 'Afternoon', 'Evening') 
        Returns:
		-------
        § p: float 
			action probability
        '''
        
        self.set_coef(self.user_type)
        self.coef_arr_up = self.coef_mat_up.loc[time]
        self.coef_arr_down = self.coef_mat_down.loc[time]
        
        self.p_up  = 1/(np.exp((-1)*(    
                       self.coef_arr_up[0] + \
                       RHin*self.coef_arr_up[1] + \
                       Tout*self.coef_arr_up[2] + \
                       Solarrad*self.coef_arr_up[3] + \
                       Windsp*self.coef_arr_up[4])) + 1)

        self.p_down  = 1/(np.exp((-1)*(
                           self.coef_arr_down[0] + \
                           RHin*self.coef_arr_down[1] + \
                           Tout*self.coef_arr_down[2] + \
                           Solarrad*self.coef_arr_down[3] + \
                           Windsp*self.coef_arr_down[4])) + 1)
                        
        return [self.p_up, self.p_down]       
        
    
    def update_sp(self, RHin, Tout, Solarrad, Windsp, time = 'Day', occ=1):
        '''  
        Update the state of the window by comparing the action 
        probability with a random number from 0 to 1.         
        Parameters:
        ----------
        Predict the probability of an action taking place        
        Parameters:
        ----------
        § RHin: float 
			Indoor relative humidity in % (i.e. 50, not 0.5)
        § Tout: float 
			Outdoor temperature in °C
        § Solarrad: float  
            Solar hours in h
		§ Windsp: float 
			Wind speed in m/s
        § time: string (default = 'Day')
			Time of the day:
                - ('Night', 'Morning', 'Day', 'Afternoon', 'Evening') 
        § occ: integer (default = 1) 
            Occupancy status (1 = Present, 0 = Absent)
        '''
        
        self.predict_proba(RHin, Tout, Solarrad, Windsp, time)
        
        self.SP = self.SP + np.greater(self.p_up, np.random.rand())*occ - \
                    np.greater(self.p_down, np.random.rand())*occ
                    
        if self.SP > self.max_SP:
            self.SP = self.max_SP
            
        if self.SP < self.min_SP:
            self.SP = self.min_SP
            
     
if __name__ == '__main__':
    pass

