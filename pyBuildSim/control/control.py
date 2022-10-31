# -*- coding: utf-8 -*-

"""
author: Nicolas Carbonare
mail: nicocarbonare@gmail
last updated: June 06, 2021
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

from .fuzzy_functions import *

    
class Controller_openloop:
    '''TO DO
    Open-loop controller for demand response (DR) systems. 
    Create first and array of controlled variable, and then an   
    array of speed levels. The algorithm calculated the speed   
    value assigned to a measurement.
    WARNING: the length of the array of speeds MUST be greather than  
    the length of the array of the controlled variable in one. 	   
    '''
    def __init__(self, **kwargs):
        '''
        Allowed keyword arguments (**kwargs):
		------------------------------------
		§ var_cont: 1d array of floats (default = [600, 800, 1400])
				Step values for fan speed
		§ speed_ar: 1d array of floats (default = [1000, 2000, 2800, 3600])
				Step values of the controlled variable 
		§ quad_targ: float (default = 0.40)
				value for the quadratic control where it sets its minimum value                            
		§ quad_amp: float (default = 0.05)
				hysteresis value for quadratic control
		§ quad_a: float (default = 10.0)
				coefficient "a" for quadratic control
		§ quad_b: float (default = 0.0)
				coefficient "b" or intercept for quadratic control
		§ quad_lowlim: float (default = 0.40)
				value to limit the quadratic control to the	upper half of the equation
		§ log_a: float (default = 2.20)
				coefficient "a" or intercept for logit control
		§ log_b: float (default = -1.10)
				coefficient "b" or intercept for logit control
		§ log_c: float (default = 300.0)
				coefficient "c" or intercept for logit control
		§ log_d: float (default = 405.0)
				coefficient "d" or intercept for logit control
        '''
        allowed_keys = {'var_cont', 'speed_ar', 'quad_targ',
                        'quad_amp','quad_a','quad_b','quad_lowlim',
                        'log_a', 'log_b','log_c','log_d'}

        rejected_keys = set(kwargs.keys()) - set(allowed_keys)
        if rejected_keys:
            raise ValueError("Invalid arguments in constructor: {}".
                             format(rejected_keys))
        else:
            self.default_values()
            self.__dict__.update((k, v) for k, v in kwargs.items() 
            if k in allowed_keys)
            self.check_steps()
    
    def default_values(self):
        '''
        Reset the values to the default ones
        '''
        self.var_cont           = [600, 800, 1400]
        self.speed_ar           = [1000.0, 2000.0, 2800.0, 3600.0]
        self.level              = 0
        self.output             = self.speed_ar[self.level]
        self.quad_targ          = 0.40
        self.quad_amp           = 0.05
        self.quad_a             = 10.0
        self.quad_b             = 0.0
        self.quad_lowlim        = 0.40
        self.log_a              = 2.20
        self.log_b              = -1.10
        self.log_c              = 320.0
        self.log_d              = 450.0
        
        
    def check_steps(self):
        '''
        Check the number of steps and associated fan speeds 
        '''
                
        if np.not_equal((len(self.speed_ar) - len(self.var_cont)), 1):
            print('ERROR: The number of speeds and '+ 
                  'thresholds do not correspond')
    
    
    def Steps_manual(self, level):     
        '''
        Calculate the step according to the measured variable
		Parameters:
		----------
		§ var_mes: float
				instantaneous measurement of the variable
        '''
        level = np.clip(level,0,len(self.speed_ar)-1)
        self.output = self.speed_ar[level]

        return self.output 
    
    def steps(self, var_mes):     
        '''
        Calculate the step according to the measured variable
		Parameters:
		----------
		§ var_mes: float
				instantaneous measurement of the variable
        '''
        self.output = self.speed_ar[sum(np.greater(var_mes, self.var_cont))]

        return self.output 
        
        
    def linear(self, var_mes):     
        '''
		The output of the control system is calculated as a linear function:
			y = a*var_mes + b
		Parameters:
		----------
		§ var_mes: float
				instantaneous measurement of the variable        
		Returns:
		--------
		§ output: float
				RPM of the fan
		'''
        self.a = (self.speed_ar[(len(self.speed_ar)-1)] - self.speed_ar[0])/ \ #change self.speed_ar[(len(self.speed_ar)-1)] for self.speed_ar[-1] and check!  
                 (self.var_cont[(len(self.var_cont)-1)] - self.var_cont[0])
        self.b = self.speed_ar[0]
        self.output = self.a*min(max(0,var_mes - self.var_cont[0]),
                                 self.var_cont[(len(self.var_cont)-1)] - \
                                 self.var_cont[0]) + \
                      self.b 

        return self.output 


    def quad(self, var_mes):     
        '''
		The output of the control system is calculated as a quadratic function:
			y = a*var_mes^2 + b
		Parameters:
		----------
		§ var_mes: float
				instantaneous measurement of the variable        
		Returns:
		--------
		§ output: float
				RPM of the fan
		'''
        if var_mes > 1:
            var_mes = var_mes/100.0
            
        self.var_lim = max(self.quad_lowlim, var_mes)
        self.x = max(0, abs(self.var_lim - self.quad_targ)-self.quad_amp)
        self.func = self.quad_a*(self.x)**2 + self.quad_b 
        self.output = np.clip(self.func,0,1)* \
                    (self.speed_ar[(len(self.speed_ar)-1)] - \
                                   self.speed_ar[0]) + \
                    self.speed_ar[0]

        return self.output         


    def logit(self, var_mes):     
        '''
		The output of the control system is calculated as an exponential function:
		    y = a/[1+e^(1-(var_mes-d)/c)] + b
		Parameters:
		----------
		§ var_mes: float
				instantaneous measurement of the variable   
		Returns:
		--------
		§ output: float
				RPM of the fan
		'''
        self.e_x = np.exp(1 - (var_mes - self.log_d)/self.log_c)
        self.func = self.log_a/(1 + self.e_x) + self.log_b 
        self.output = np.clip(self.func,0,1)*(self.speed_ar[(len(self.speed_ar)-1)] - \
                                              self.speed_ar[0]) + \
                             self.speed_ar[0]

        return self.output 


    def cf_RH_CO2(self, var_mes_1, var_mes_2):     
        '''
		The output of the control system is calculated as the maximum value 
        of the RH and CO2 functions
		    y = a/[1+e^(1-(var_mes-d)/c)] + b
		Parameters:
		----------
		§ var_mes_1: float
				instantaneous measurement of the variable 1 
		§ var_mes_2: float
				instantaneous measurement of the variable 2 
        Returns:
		--------
		§ output: float
				RPM of the fan
		'''
        self.cf                 = cost_func()
        self.RH                 = self.cf.RH(var_mes_1)
        self.CO2                = self.cf.CO2(var_mes_2)      
        self.values             = [self.RH, self.CO2]
        
        #self.ind                = max(xrange(len(self.values)), 
        #                              key=self.values.__getitem__) # python 2.7
        self.ind                = max(range(len(self.values)), 
                                      key=self.values.__getitem__) #python 3.7

        self.output             = self.Quad(var_mes_1)*\
                                            np.equal(self.ind, 0) + \
                                    self.Logit(var_mes_2)*\
                                            np.equal(self.ind, 1)
                                            
        return self.output 


class Controller_closedloop:
    '''
    controller based on sigmoid (logistic) function
    Author Felix Ohr @fohr
    '''
    def __init__(self, setvalue=293.15, 
                       prange=2.0, 
                       a = 10.0, 
                       minmax=[0.0, 1.0]):
        self.setvalue           = setvalue
        self.prange             = prange
        self.a                  = a
        self.min                = min(minmax)
        self.max                = max(minmax)

    def sigmoid(self, feedback, pr = False):
        self.feedback           = feedback
        self.output             =  1/(1+np.exp((self.a/self.prange)*(self.feedback - self.setvalue)))
        self.output             = (self.max - self.min)*self.output + self.min

        if pr:
            return self.output


class PIDHysteresis:
    '''
    PID or OnOff controller with hysteresis and min/max bounds
    Author Felix Ohr @fohr
    '''
    def __init__(self, kp = 0.2,
                       ki = 0.0, 
                       kd = 0.0,
                       onoff = True,
                       currenttime = 0.0, 
                       sampletime  = 1.0,
                       hyst_high  = 5.0,
                       hyst_low  = -5.0,
                       minbound = 0.0,
                       maxbound = 1.0,
                       windup = 5.0,
                       ):
        '''
        kp - P-gain
        ki - I-gain
        kd - D-gain
        currenttime - current time in seconds or same time scale as sample time
        sampletime  - sample time of discrete PID control in seconds or 
            same time scale as current time
        minbound - smallest possible output value
        maxbound - gretest possible output value
        '''
        self.clear()
        self.reset(currenttime, sampletime)
        self.set_parameters(kp, ki, kd)
        self.set_hysteresis(hyst_high, hyst_low, hystflag=True)
        self.minbound           = minbound
        self.maxbound           = maxbound
        self.set_windup(windup)
        self.onoff              = onoff

    def clear(self):
        '''
        clears PID computations and coefficients
        '''
        self.setpoint           = 0.0
        self.PTerm              = 0.0
        self.ITerm              = 0.0
        self.DTerm              = 0.0
        self.last_error         = 0.0
        self.int_error          = 0.0
        self.delta_error        = 0.0
        self.windup             = 5.0
        self.output             = 0.0


    def reset(self, currenttime, sampletime):
        '''
        sample time PID that should be updated at a regular interval.
        Based on a pre-determined sampe time, the PID decides if it 
        should compute or return immediately.
        '''
        self.currenttime        = currenttime
        self.lasttime           = currenttime
        self.deltatime          = 0.0
        self.sampletime         = sampletime


    def update(self, feedback, currenttime, pr=False):
        '''
        Calculates PID value for given reference feedback
        '''
        self.error              = self.setpoint - feedback
        self.currenttime        = currenttime
        self.deltatime          = self.currenttime - self.lasttime
        self.delta_error        = self.error - self.last_error

        if self.hystflag and np.less_equal(self.error, -self.hyst_high):
            self.hystflag       = False
        elif not(self.hystflag) and np.greater_equal(self.error, -self.hyst_low):
            self.hystflag       = True
        else:
            self.hystflag       = self.hystflag
            
        if (self.deltatime >= self.sampletime):
            self.PTerm          = self.kp * self.error
            self.ITerm         += self.error * self.deltatime

            if (self.ITerm < -self.windup):
                self.ITerm      = -self.windup
            elif (self.ITerm > self.windup):
                self.ITerm      = self.windup

            self.DTerm = 0.0
            if self.deltatime > 0:
                self.DTerm      = self.delta_error / self.deltatime

            # remember last time and last error for next calculation
            self.lasttime       = self.currenttime
            self.last_error     = self.error
            self.output         = 1.0*self.onoff*self.hystflag + \
                                 (1.0 - self.onoff)*self.hystflag*(self.PTerm + \
                                                                   (self.ki * self.ITerm) + \
                                                                   (self.kd * self.DTerm))

            self.output         = max(self.minbound, min(self.maxbound, self.output))
        
        if pr: 
            return self.output
            
            
    def set_parameters(self, proportional_gain, integral_gain, derivative_gain):
        '''
        proportional_gain   - determines how aggressively the PID reacts to 
            the current error by setting proportional gain
        integral_gain       - determines how aggressively the PID reacts to 
            the current error by setting integral gain
        derivative_gain     - determines how aggressively the PID reacts to 
            the current error by setting derivative gain
        '''
        self.kp                 = proportional_gain
        self.ki                 = integral_gain
        self.kd                 = derivative_gain

    def set_hysteresis(self, hyst_high, hyst_low, hystflag):
        self.hyst_high          = max(hyst_high, 0.0)
        self.hyst_low           = min(hyst_low, 0.0)
        self.hystflag           = hystflag

    def set_windup(self, windup):
        '''
        Integral windup, also known as integrator windup or reset windup,
        refers to the situation in a PID feedback controller where
        a large change in setpoint occurs (say a positive change)
        and the integral terms accumulates a significant error
        during the rise (windup), thus overshooting and continuing
        to increase as this accumulated error is unwound
        (offset by errors in the other direction).
        The specific problem is the excess overshooting.
        '''
        self.windup             = windup

#-------------------------------------------------------------------------------# 
class Controller_Ventilation_Fuzzy_Summer:  
    '''
    --------------------------------------------------------------
    Fuzzy logic controller taking into account two variables Troom and Tamb 
    for the summer control of residential ventilation systems.
    Assumes a continuous control field for the fan speed
    Assumes a sigmoid membership function for the controlled variables 
    (Troom and Tamb) and a linear membership function for the output of the 
    controller (fan speed). The membership functions have three categories.
    --------------------------------------------------------------              
    '''
    def __init__(self, 
                 fanspeed_ar        = [800.0, 3600.0],
                 Troom_range        = np.arange(12, 30.5, 0.5),
                 Tamb_range         = np.arange(12, 30.5, 0.5),
                 fan_range          = np.arange(0, 1.01, 0.01),
                 Troom_std          = 1.5,
                 Troom_lo_b         = 18,
                 Troom_hi_b         = 26,
                 Tamb_std           = 1.5,
                 Tamb_lo_b          = 16,
                 Tamb_hi_b          = 26,
                 fan_lo             = 0.33, 
                 fan_hi             = 0.67 
                 ):
        '''
        Parameters:
            § speed_ar = array-like with the extreme values for fan speed
            § Troom_range = array with the range of relative humidity values covered
                        by the control strategy
            § Tamb_range = array with the range of Tamb values covered by the 
                        control strategy
            § fan_range = array with the range of fan speed values covered by
                        the control strategy
            § Troom_c = coefficient of the sigmoid membership function for Troom
            § Troom_lo_b = Troom value where the membership functions cross each other
                        on the lower side
            § Troom_hi_b = Troom value where the membership functions cross each other
                        on the upper side
            § Troom_min = minimum admisible value for Troom. Triggers special control
            § Troom_max = maximum admisible value for Troom. Triggers special control
            § Tamb_c = coefficient of the sigmoid membership function for Tamb
            § Tamb_lo_b = Tamb value where the membership functions cross each other
                        on the lower side
            § Tamb_hi_b = Tamb value where the membership functions cross each other
                        on the upper side
            § Troom_min = minimum admisible value for Tamb. Triggers special control
            § Troom_max = maximum admisible value for Tamb. Triggers special control
            § fan_lo = fan speed percentage value where the membership functions 
                        cross each other on the lower side
            § fan_hi = fan speed percentage value where the membership functions 
                        cross each other on the upper side
        -------------------------------------------------------------- 
        ''' 
        
        self.fanspeed_ar            = fanspeed_ar
        self.Troom_range            = Troom_range
        self.Tamb_range             = Tamb_range
        self.fan_range              = fan_range
        self.Troom_std              = Troom_std
        self.Troom_lo_b             = Troom_lo_b
        self.Troom_hi_b             = Troom_hi_b
        self.Tamb_std               = Tamb_std
        self.Tamb_lo_b              = Tamb_lo_b
        self.Tamb_hi_b              = Tamb_hi_b
        self.fan_lo                 = fan_lo
        self.fan_hi                 = fan_hi

        self.fuzz                   = fuzzy_func()

   
    def membership_func_gauss_5(self, var_range, var_low, var_high, std):
        '''
        Definition of a gaussian membership function
        Parameters:
            § var_range = array with the range of the variable values covered
                        by the control strategy
            § var_vlow = var value where the membership function reaches 1
            § var_low = var value where the membership function reaches 1
            § var_med = var value where the membership function reaches 1
            § var_high = var value where the membership function reaches 1
            § std = var standard deviation value, where the membership
                    functions cross each other
        '''
        mean_med = (var_high + var_low)/2   
        std_med = (var_high - var_low)/4/std  
        var_lo_memb = self.fuzz.gaussmf(var_range, var_low, std)
        var_hi_memb = self.fuzz.gaussmf(var_range, var_high, std)
        var_md_memb = self.fuzz.gaussmf(var_range, mean_med, std_med)
        
        var_vlo_memb = 1 - self.fuzz.sigmf(var_range, var_low-1.177*std, std)
        var_vhi_memb = self.fuzz.sigmf(var_range, var_high+1.177*std, std)
        
        # if np.greater_equal((var_low - std), min(var_range)):
        #     var_vlo_memb[(var_range < (var_low - std))] = 1.
        # if np.greater_equal(max(var_range), (var_high) + std):
        #     var_vhi_memb[(var_range > (var_high + std))] = 1.
            
        return [var_vlo_memb, var_lo_memb, var_md_memb, 
                var_hi_memb, var_vhi_memb]

    def membership_func_lin_3(self, var_range, var_low, var_high):
        '''
        Definition of a linear membership function
        Parameters:
            § var_range = array with the range of the variable values covered
                        by the control strategy
            § var_low = var value where the membership functions cross each other
                        on the lower side
            § var_high = var value where the membership functions cross each other
                        on the upper side
        '''                         
        band = (max(var_range)-min(var_range))*5 / (len(var_range)-1)
        
        # Generate fuzzy membership functions
        var_lo_memb = self.fuzz.trimf(var_range, [var_low - band, var_low -band, var_low + band])
        var_md_memb = self.fuzz.trapmf(var_range, [var_low - band, var_low + band, 
                                            var_high - band, var_high + band])
        var_hi_memb = self.fuzz.trimf(var_range, [var_high - band, var_high + band, var_high + band])
                   
        if np.greater_equal((var_low - band), min(var_range)):
            var_lo_memb[(var_range < (var_low - band))] = 1.
        if np.greater_equal(max(var_range), (var_high) + band):
            var_hi_memb[(var_range > (var_high) + band)] = 1.
            
        return [var_lo_memb, var_md_memb, var_hi_memb]
    
    
    def plot_membership(self, save=False):
        '''
        Generate and plot the membership functions for all variables
        '''
        colors =  ['#179C7D', '#1F82C0', '#E2001A', 
                   '#F29400', '#39378B']
        # Generate fuzzy membership functions
        self.Troom_memb = self.membership_func_gauss_5(self.Troom_range,
                                                       self.Troom_lo_b, 
                                                       self.Troom_hi_b,
                                                       self.Troom_std) 
        self.Tamb_memb = self.membership_func_gauss_5(self.Tamb_range,
                                                      self.Tamb_lo_b, 
                                                      self.Tamb_hi_b,
                                                      self.Tamb_std)
        self.fan_memb = self.membership_func_lin_3(self.fan_range,
                                                   self.fan_lo,
                                                   self.fan_hi)  
        
        fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(7, 9))
        plt.subplots_adjust(bottom = 0.06, top = 0.91,
                            left = 0.09, right = 0.95) 
            
        ax0.plot(self.Troom_range, self.Troom_memb[0], colors[0], linewidth=2, label='Too cold')
        ax0.plot(self.Troom_range, self.Troom_memb[1], colors[1], linewidth=2, label='Slightly cold')
        ax0.plot(self.Troom_range, self.Troom_memb[2], colors[2], linewidth=2, label='Comfortable')
        ax0.plot(self.Troom_range, self.Troom_memb[3], colors[3], linewidth=2, label='Slightly warm')
        ax0.plot(self.Troom_range, self.Troom_memb[4], colors[4], linewidth=2, label='Too warm')
        ax0.set_title('Room temperature [$^{\circ} C$]',fontsize = 13)
        ax0.legend(fontsize = 12, loc = 'center right')
        ax0.set_xlim([min(self.Troom_range), max(self.Troom_range)])
        
        ax1.plot(self.Tamb_range, self.Tamb_memb[0], colors[0], linewidth=2, label='Very cold')
        ax1.plot(self.Tamb_range, self.Tamb_memb[1], colors[1], linewidth=2, label='Cold')
        ax1.plot(self.Tamb_range, self.Tamb_memb[2], colors[2], linewidth=2, label='Neutral')
        ax1.plot(self.Tamb_range, self.Tamb_memb[3], colors[3], linewidth=2, label='Warm')
        ax1.plot(self.Tamb_range, self.Tamb_memb[4], colors[4], linewidth=2, label='Very warm')
        ax1.set_title('Ambient temperature [$^{\circ} C$]',fontsize = 13)
        ax1.legend(fontsize = 13, loc = 'center right')
        ax1.set_xlim([min(self.Tamb_range), max(self.Tamb_range)])
        
        ax2.plot(self.fan_range*100, self.fan_memb[0], colors[0], linewidth=2, label='Low')
        ax2.plot(self.fan_range*100, self.fan_memb[1], colors[1], linewidth=2, label='Medium')
        ax2.plot(self.fan_range*100, self.fan_memb[2], colors[2], linewidth=2, label='High')
        ax2.set_title('Fan speed [%]',fontsize = 13)
        ax2.legend(fontsize = 13, loc = 'center right')
        ax2.set_xlim([0,100])
        
        # Turn off top/right axes
        for ax in (ax0, ax1, ax2):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
            ax.tick_params(labelsize = 13)

        plt.tight_layout()
        if save:
            fig.savefig('fuzzy_memb.svg',format='svg',dpi=600)
            fig.savefig('fuzzy_memb.png',format='png',dpi=300)
            fig.savefig('fuzzy_memb.pdf',format='pdf',dpi=600)


    def interpret_membership(self, Troom, Tamb):
        '''
        Interpretes the membership fan function with the measured values 
        of Troom & Tamb and their respective membership functions.
        Returns as output the percentage of fan after defuzzification.
        Parameters:
		----------
		§ Troom: float
 			  Measured Troom value
		§ Tamb: float
    		  Measured Tamb value
		'''
        self.Troom_clip = np.clip(Troom,min(self.Troom_range),max(self.Troom_range))
        self.Tamb_clip = np.clip(Tamb,min(self.Tamb_range),max(self.Tamb_range))
        
        #Definition of membership function
        self.Troom_memb = self.membership_func_gauss_5(self.Troom_range,
                                                     self.Troom_lo_b, 
                                                     self.Troom_hi_b,
                                                     self.Troom_std) 
        self.Tamb_memb = self.membership_func_gauss_5(self.Tamb_range,
                                                    self.Tamb_lo_b, 
                                                    self.Tamb_hi_b,
                                                    self.Tamb_std)
        self.fan_memb = self.membership_func_lin_3(self.fan_range,
                                                   self.fan_lo,
                                                   self.fan_hi)  
        
        #Intepretation of membership functions
        self.Troom_level_vlo = self.fuzz.interp_membership(self.Troom_range, 
                                                     self.Troom_memb[0], 
                                                     self.Troom_clip)
        self.Troom_level_lo = self.fuzz.interp_membership(self.Troom_range, 
                                                    self.Troom_memb[1], 
                                                    self.Troom_clip)
        self.Troom_level_md = self.fuzz.interp_membership(self.Troom_range, 
                                                    self.Troom_memb[2], 
                                                    self.Troom_clip)
        self.Troom_level_hi = self.fuzz.interp_membership(self.Troom_range, 
                                                    self.Troom_memb[3], 
                                                    self.Troom_clip)
        self.Troom_level_vhi = self.fuzz.interp_membership(self.Troom_range, 
                                                     self.Troom_memb[4], 
                                                     self.Troom_clip)
 
        self.Tamb_level_vlo = self.fuzz.interp_membership(self.Tamb_range, 
                                                     self.Tamb_memb[0], 
                                                     self.Tamb_clip)       
        self.Tamb_level_lo = self.fuzz.interp_membership(self.Tamb_range, 
                                                    self.Tamb_memb[1], 
                                                    self.Tamb_clip)
        self.Tamb_level_md = self.fuzz.interp_membership(self.Tamb_range, 
                                                    self.Tamb_memb[2], 
                                                    self.Tamb_clip)
        self.Tamb_level_hi = self.fuzz.interp_membership(self.Tamb_range, 
                                                    self.Tamb_memb[3], 
                                                    self.Tamb_clip)
        self.Tamb_level_vhi = self.fuzz.interp_membership(self.Tamb_range, 
                                                     self.Tamb_memb[4], 
                                                     self.Tamb_clip)
        
        #### Fuzzy rules ###
        ####
        # Fan speed	       Tamb
        # 	        VC	C	N	W	VW
        #    	TC	L	L	M	M	M
        # 	    SC	L	L	M	M	M
        # Troom Cf	M	M	M	L	L
        # 	    SW	H	H	H	L	L
        # 	    TW	H	H	H	L	L


        ## Fan low
        # Tamb VC or C
        self.active_rulel_ax1 = np.fmax(self.Tamb_level_vlo, self.Tamb_level_lo)     
        # Tamb VW or W
        self.active_rulel_ax2 = np.fmax(self.Tamb_level_vhi, self.Tamb_level_hi)
        # Troom TC or C
        self.active_rulel_ax3 = np.fmax(self.Troom_level_vlo, self.Troom_level_lo)
        # Troom Cf or W or SW
        self.active_rulel_ax4 = np.fmax(self.Troom_level_md, np.fmax(self.Troom_level_hi, 
                                                                     self.Troom_level_vhi))
        
        # Tamb ax1 and Troom ax3
        self.active_rulel_1 = np.fmin(self.active_rulel_ax1, self.active_rulel_ax3)
        # Tamb ax2 and Troom ax4
        self.active_rulel_2 = np.fmin(self.active_rulel_ax2, self.active_rulel_ax4)
        # Active rule 1 or 2
        self.active_rulel = np.fmax(self.active_rulel_1, self.active_rulel_2)
        self.fan_activation_lo = np.fmin(self.active_rulel, self.fan_memb[0])  

        ## Fan med
        # Tamb VC or C or N
        self.active_rulem_ax1 = np.fmax(self.Tamb_level_vlo, np.fmax(self.Tamb_level_lo, 
                                                                     self.Tamb_level_md)) 
        # Tamb VW or W or N
        self.active_rulem_ax2 = np.fmax(self.Tamb_level_vhi, np.fmax(self.Tamb_level_hi, 
                                                                     self.Tamb_level_md))
        # Troom TC or C or N
        self.active_rulem_ax3 = np.fmax(self.Troom_level_vlo, np.fmax(self.Troom_level_md,
                                                                      self.Troom_level_lo))
        
        # Tamb ax1 and Troom Cf
        self.active_rulem_1 = np.fmin(self.active_rulem_ax1, self.Troom_level_md)
        # Tamb ax2 and Troom SC
        self.active_rulem_2 = np.fmin(self.active_rulem_ax2, self.Troom_level_lo)
        # Troom ax3 and Tamb N
        self.active_rulem_3 = np.fmin(self.active_rulem_ax3, self.Tamb_level_md)
        # Active rule 1 or 2 or 3
        self.active_rulem = np.fmax(self.active_rulem_1, np.fmax(self.active_rulem_2,
                                                                 self.active_rulem_3))
        self.fan_activation_md = np.fmin(self.active_rulem, self.fan_memb[1])  

        ## Fan high
        # Tamb VC or C or N
        self.active_ruleh_ax1 = np.fmax(self.Tamb_level_vlo, np.fmax(self.Tamb_level_lo, 
                                                                     self.Tamb_level_md)) 
        # Tamb VW or W
        self.active_ruleh_ax2 = np.fmax(self.Tamb_level_vhi, self.Tamb_level_hi)
        # Troom TW or SW
        self.active_ruleh_ax3 = np.fmax(self.Troom_level_vhi, self.Troom_level_hi)
        
        # Tamb ax1 and Troom ax3
        self.active_ruleh_1 = np.fmin(self.active_ruleh_ax1, self.active_ruleh_ax3)
        # Tamb ax2 and Troom TC
        self.active_ruleh_2 = np.fmin(self.active_ruleh_ax2, self.Troom_level_vlo)
        # Active rule 1 or 2 or 3
        self.active_ruleh = np.fmax(self.active_ruleh_1, self.active_ruleh_2)
        self.fan_activation_hi = np.fmin(self.active_ruleh, self.fan_memb[2])  
        
        # Aggregated functions
        self.aggregated = np.fmax(self.fan_activation_lo,
                                  np.fmax(self.fan_activation_md, 
                                          self.fan_activation_hi))
        
        # Fan speed calculation (fan_output)       
        self.fan_defuzz = self.fuzz.defuzz_centroid(self.fan_range, 
                                                    self.aggregated) 
        return self.fan_defuzz
    
    
    def output(self, Troom, Tamb, occ=1, plot=False):
        '''
        Calculate the output of the controller (fan speed) as a consequence 
        of the measurements of Troom and Tamb
        '''
        
        # Inteprete membership functions
        self.fan_defuzz = self.interpret_membership(Troom,Tamb)
        
        # Fan speed calculation (fan_output)   
        if occ==1:
            if (((Tamb > max(self.Tamb_range)) &
                  (Troom > min(self.Troom_range))) | (Troom > max(self.Troom_range))):
                self.fan_output = self.fanspeed_ar[1]
            elif(Troom <= min(self.Troom_range)):
                self.fan_output = self.fanspeed_ar[0]    
            else:
                self.fan_output = self.fan_defuzz*self.fanspeed_ar[1]*\
                                    np.greater_equal(self.fan_defuzz*self.fanspeed_ar[1],
                                                self.fanspeed_ar[0]) + \
                                    self.fanspeed_ar[0]*np.greater(self.fanspeed_ar[0],
                                                self.fan_defuzz*self.fanspeed_ar[1])
        else:
                self.fan_output = self.fanspeed_ar[0]        
                
        if plot:
            fan_activation = self.fuzz.interp_membership(self.fan_range, 
                                                         self.aggregated, 
                                                         self.fan_defuzz)

            fig, ax0 = plt.subplots(figsize=(8, 3))
            ax0.plot(self.fan_range, self.fan_memb[0], 'b', linewidth=0.5, linestyle='--')
            ax0.plot(self.fan_range, self.fan_memb[1], 'g', linewidth=0.5, linestyle='--')
            ax0.plot(self.fan_range, self.fan_memb[2], 'r', linewidth=0.5, linestyle='--')
            ax0.fill_between(self.fan_range, np.zeros_like(self.fan_range), 
                              self.aggregated, facecolor='Orange', alpha=0.7)
            ax0.plot([self.fan_defuzz, self.fan_defuzz], 
                      [0, fan_activation], 'k', linewidth=1.5, alpha=0.9)
            ax0.set_title('Aggregated membership and result (line)')
            
            # Turn off top/right axes
            for ax in (ax0,):
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.get_xaxis().tick_bottom()
                ax.get_yaxis().tick_left()

            plt.tight_layout()

        return self.fan_output        
    
    def plot_output_control(self, save=False):
        '''
        Plots the control map in 3D
        '''
        fan_output_range = []

        #Custom colormap
        colors = ['#179C7D', '#FFDC00', '#E2001A']  # FhG: B -> G -> R
        n_bins = 100  # Discretizes the interpolation into bins
        cmap_name = 'FhG_RGB'
        cm = LinearSegmentedColormap.from_list(
                cmap_name, colors, N=n_bins)
                        
        #Intepretation of membership functions
        for j in self.Troom_range:
            for i in self.Tamb_range:
                fan_output_range = np.append(fan_output_range,
                                              self.interpret_membership(j,i))
                
        #from matplotlib import cm
        xx, yy = np.meshgrid(self.Tamb_range, self.Troom_range)       
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        plt.subplots_adjust(bottom = 0.08, top = 0.96,
                        left = 0.04, right = 0.92) 
        ax.plot_surface(xx, yy, 
                        fan_output_range.reshape(len(self.Troom_range),
                                                  len(self.Tamb_range))*100, 
                        cmap=cm , rstride=1, cstride=1,
                        antialiased=True)
#        ax.zaxis.set_ticks(np.arange(0.2,0.9,0.1))
        ax.set_ylabel('Room temperature [$^{\circ}C$]',fontsize = 13,labelpad=10)                
        ax.set_xlabel('Ambient temperature [$^{\circ}C$]',fontsize = 13,labelpad=10)                
        ax.set_zlabel('Fan speed [%]',fontsize = 13,labelpad=10)                
        ax.tick_params(labelsize = 10, axis='both')                
                        
        if save:
            fig.savefig('fuzzy_output.svg',format='svg',dpi=600)
            fig.savefig('fuzzy_output.png',format='png',dpi=300)
            fig.savefig('fuzzy_output.pdf',format='pdf',dpi=600)



#-------------------------------------------------------------------------------#
class Controller_Ventilation_Fuzzy_Winter:  
    '''
    --------------------------------------------------------------
    Fuzzy logic controller taking into account two variables RH and CO2 
    for winter control of residential ventilation systems
    Assumes a continuous control field for the fan speed
    Assumes a sigmoid membership function for the controlled variables 
    (RH and CO2) and a linear membership function for the output of the 
    controller (fan speed). The membership functions have three categories.
    --------------------------------------------------------------              
    '''     
    def __init__(self, 
                  fanspeed_ar        = [800.0, 3600.0],
                  RH_range           = np.arange(0, 1.02, 0.02),
                  CO2_range          = np.arange(400, 1820., 20),
                  fan_range          = np.arange(0, 1.01, 0.01),
                  RH_c               = 25.,
                  RH_lo_b            = 0.30,
                  RH_hi_b            = 0.70,
                  RH_min             = 0.25,
                  RH_max             = 0.75,
                  CO2_c              = 0.02,
                  CO2_lo_b           = 800.,
                  CO2_hi_b           = 1250.,
                  fan_lo             = 0.33, 
                  fan_hi             = 0.67 
                  ):
        '''
        Parameters:
            § speed_ar = array-like with the extreme values for fan speed
            § RH_range = array with the range of relative humidity values covered
                        by the control strategy
            § CO2_range = array with the range of CO2 values covered by the 
                        control strategy
            § fan_range = array with the range of fan speed values covered by
                        the control strategy
            § RH_c = coefficient of the sigmoid membership function for RH
            § RH_lo_b = RH value where the membership functions cross each other
                        on the lower side
            § RH_hi_b = RH value where the membership functions cross each other
                        on the upper side
            § RH_min = minimum admisible value for RH. Triggers special control
            § RH_max = maximum admisible value for RH. Triggers special control
            § CO2_c = coefficient of the sigmoid membership function for CO2
            § CO2_lo_b = CO2 value where the membership functions cross each other
                        on the lower side
            § CO2_hi_b = CO2 value where the membership functions cross each other
                        on the upper side
            § fan_lo = fan speed percentage value where the membership functions 
                        cross each other on the lower side
            § fan_hi = fan speed percentage value where the membership functions 
                        cross each other on the upper side
        -------------------------------------------------------------- 
        ''' 
        
        self.fanspeed_ar            = fanspeed_ar
        self.RH_range               = RH_range
        self.CO2_range              = CO2_range
        self.fan_range              = fan_range
        self.RH_c                   = RH_c
        self.RH_lo_b                = RH_lo_b
        self.RH_hi_b                = RH_hi_b
        self.RH_min                 = RH_min
        self.RH_max                 = RH_max
        self.CO2_c                  = CO2_c
        self.CO2_lo_b               = CO2_lo_b
        self.CO2_hi_b               = CO2_hi_b
        self.fan_lo                 = fan_lo
        self.fan_hi                 = fan_hi
        
        self.fuzz                   = fuzzy_func()


    def membership_func_lin(self, var_range, var_low, var_high):
        '''
        Definition of a linear membership function
        Parameters:
            § var_range = array with the range of the variable values covered
                        by the control strategy
            § var_low = var value where the membership functions cross each other
                        on the lower side
            § var_high = var value where the membership functions cross each other
                        on the upper side
        '''                         
        band = (max(var_range)-min(var_range))*5 / (len(var_range)-1)
        
        # Generate fuzzy membership functions
        var_lo_memb = self.fuzz.trimf(var_range, [var_low - band, var_low -band, var_low + band])
        var_md_memb = self.fuzz.trapmf(var_range, [var_low - band, var_low + band, 
                                                   var_high - band, var_high + band])
        var_hi_memb = self.fuzz.trimf(var_range, [var_high - band, var_high + band, var_high + band])
                   
        if np.greater_equal((var_low - band), min(var_range)):
            var_lo_memb[(var_range < (var_low - band))] = 1.
        if np.greater_equal(max(var_range), (var_high) + band):
            var_hi_memb[(var_range > (var_high) + band)] = 1.
            
        return [var_lo_memb, var_md_memb, var_hi_memb]
    
    
    def membership_func_sigm(self, var_range, var_c, var_b_lo, var_b_hi):
        '''
        Definition of a sigmoid membership function
        Parameters:
            § var_range = array with the range of the variable (var) values 
                        covered by the control strategy
            § var_c = coefficient of the sigmoid membership function for the var
            § var_b_lo = var value where the membership functions cross each 
                        other on the lower side
            § var_b_hi = var value where the membership functions cross each 
                        other on the upper side
        '''                                          
        # Generate fuzzy membership functions
        var_lo_memb = self.fuzz.sigmf(var_range, var_b_lo, -var_c)
        var_hi_memb = self.fuzz.sigmf(var_range, var_b_hi, var_c)
        var_md_memb = np.ones_like(var_range) - var_lo_memb - var_hi_memb
    
        return [var_lo_memb, var_md_memb, var_hi_memb]
    
    
    def plot_membership(self, save=False):
        '''
        Generate and plot the membership functions for all variables
        '''
        colors = ['#179C7D', '#F29400', '#E2001A']
                  
        # Generate fuzzy membership functions
        self.RH_memb = self.membership_func_sigm(self.RH_range, self.RH_c,
                                                 self.RH_lo_b, self.RH_hi_b) 
        self.CO2_memb = self.membership_func_sigm(self.CO2_range, self.CO2_c,
                                                  self.CO2_lo_b, self.CO2_hi_b)  
        self.fan_memb = self.membership_func_lin(self.fan_range,
                                                 self.fan_lo,
                                                 self.fan_hi)  
        
        fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(7, 9))
        plt.subplots_adjust(bottom = 0.10, top = 0.95,
                            left = 0.09, right = 0.95) 
            
        ax0.plot(self.RH_range*100, self.RH_memb[0], colors[0], linewidth=2, label='Too low')
        ax0.plot(self.RH_range*100, self.RH_memb[1], colors[1], linewidth=2, label='Acceptable')
        ax0.plot(self.RH_range*100, self.RH_memb[2], colors[2], linewidth=2, label='Too high')
        ax0.set_xlabel('Relative humidity [%]',fontsize = 13)
        ax0.legend(fontsize = 12, loc = 'center right')
        ax0.set_xlim([0,100])
        
        ax1.plot(self.CO2_range, self.CO2_memb[0], colors[0], linewidth=2, label='Excellent')
        ax1.plot(self.CO2_range, self.CO2_memb[1], colors[1], linewidth=2, label='Acceptable')
        ax1.plot(self.CO2_range, self.CO2_memb[2], colors[2], linewidth=2, label='Poor')
        ax1.set_xlabel('Indoor air quality - $CO_2$ [ppm]',fontsize = 13)
        ax1.legend(fontsize = 13)
        ax1.set_xlim([400,1800])
        
        ax2.plot(self.fan_range*100, self.fan_memb[0], colors[0], linewidth=2, label='Low')
        ax2.plot(self.fan_range*100, self.fan_memb[1], colors[1], linewidth=2, label='Medium')
        ax2.plot(self.fan_range*100, self.fan_memb[2], colors[2], linewidth=2, label='High')
        ax2.set_xlabel('Fan speed [%]',fontsize = 13)
        ax2.legend(fontsize = 13)
        ax2.set_xlim([0,100])
        
        # Turn off top/right axes
        for ax in (ax0, ax1, ax2):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
            ax.tick_params(labelsize = 13)
            ax.set_ylabel('Membership\n degree [%]', fontsize=15)

        plt.tight_layout()
        if save:
            fig.savefig('fuzzy_memb.svg',format='svg',dpi=600)
            fig.savefig('fuzzy_memb.png',format='png',dpi=300)
            fig.savefig('fuzzy_memb.pdf',format='pdf',dpi=600)


    def interpret_membership(self, RH, CO2):
        '''
        Interpretes the membership fan function with the measured values 
        of RH & CO2 and their respective membership functions.
        Returns as output the percentage of fan after defuzzification.
        Parameters:
		----------
		§ RH: float
 			  Measured RH value
		§ CO2: float
    		  Measured CO2 value
		'''
        self.RH_clip = np.clip(RH,min(self.RH_range),max(self.RH_range))
        self.CO2_clip = np.clip(CO2,min(self.CO2_range),max(self.CO2_range))
        self.RH_memb = self.membership_func_sigm(self.RH_range, self.RH_c,
                                                 self.RH_lo_b, self.RH_hi_b) 
        self.CO2_memb = self.membership_func_sigm(self.CO2_range, self.CO2_c,
                                                  self.CO2_lo_b, self.CO2_hi_b)  
        self.fan_memb = self.membership_func_lin(self.fan_range,
                                                 self.fan_lo,
                                                 self.fan_hi)  
        
        #Intepretation of membership functions
        self.RH_level_lo = self.fuzz.interp_membership(self.RH_range, 
                                                      self.RH_memb[0], 
                                                      self.RH_clip)
        self.RH_level_md = self.fuzz.interp_membership(self.RH_range, 
                                                      self.RH_memb[1], 
                                                      self.RH_clip)
        self.RH_level_hi = self.fuzz.interp_membership(self.RH_range, 
                                                      self.RH_memb[2], 
                                                      self.RH_clip)
        
        self.CO2_level_lo = self.fuzz.interp_membership(self.CO2_range, 
                                                        self.CO2_memb[0], 
                                                        self.CO2_clip)
        self.CO2_level_md = self.fuzz.interp_membership(self.CO2_range, 
                                                        self.CO2_memb[1], 
                                                        self.CO2_clip)
        self.CO2_level_hi = self.fuzz.interp_membership(self.CO2_range, 
                                                        self.CO2_memb[2], 
                                                        self.CO2_clip)
        
        #Rules of decision
        #Rule 1 - RH low OR (RH med & CO2 low) == fan low
        self.active_rule1_aux = np.fmin(self.RH_level_md, self.CO2_level_lo)
        self.active_rule1 = np.fmax(self.active_rule1_aux, self.RH_level_lo)
        self.fan_activation_lo = np.fmin(self.active_rule1, self.fan_memb[0])  

        #Rule 2 - RH med & CO2 med  == fan med
        self.active_rule2 = np.fmin(self.RH_level_md, self.CO2_level_md)
        self.fan_activation_md = np.fmin(self.active_rule2, self.fan_memb[1])  

        #Rule 3 - RH high OR (RH med & CO2 high) == fan high
        self.active_rule3_aux = np.fmin(self.RH_level_md, self.CO2_level_hi)
        self.active_rule3 = np.fmax(self.active_rule3_aux, self.RH_level_hi)
        self.fan_activation_hi = np.fmin(self.active_rule3, self.fan_memb[2])  
        
        # Aggregated functions
        self.aggregated = np.fmax(self.fan_activation_lo,
                                  np.fmax(self.fan_activation_md, 
                                          self.fan_activation_hi))
        
        # Fan speed calculation (fan_output)       
        self.fan_defuzz = self.fuzz.defuzz_centroid(self.fan_range, 
                                                    self.aggregated)
        
        return self.fan_defuzz
 
    def output(self, RH, CO2, occ=1, plot=False):
        '''
        Calculate the output of the controller (fan speed) as a consequence 
        of the measurements of RH and CO2
        '''
        
        # Inteprete membership functions
        self.fan_defuzz = self.interpret_membership(RH, CO2)
        
        # Fan speed calculation (fan_output)   
        if occ==1:
            if (((CO2 > max(self.CO2_range)) &
                  (RH > self.RH_min)) | (RH > self.RH_max)):
                self.fan_output = self.fanspeed_ar[1]
            elif(RH <= self.RH_min):
                self.fan_output = self.fanspeed_ar[0]    
            else:
                self.fan_output = self.fan_defuzz*self.fanspeed_ar[1]*\
                                    np.greater_equal(self.fan_defuzz*self.fanspeed_ar[1],
                                                self.fanspeed_ar[0]) + \
                                    self.fanspeed_ar[0]*np.greater(self.fanspeed_ar[0],
                                                self.fan_defuzz*self.fanspeed_ar[1])
        else:
                self.fan_output = self.fanspeed_ar[0]        
                
        if plot:
            fan_activation = self.fuzz.interp_membership(self.fan_range, 
                                                         self.aggregated, 
                                                         self.fan_defuzz)

            fig, ax0 = plt.subplots(figsize=(8, 3))
            ax0.plot(self.fan_range, self.fan_memb[0], 'b', linewidth=0.5, linestyle='--')
            ax0.plot(self.fan_range, self.fan_memb[1], 'g', linewidth=0.5, linestyle='--')
            ax0.plot(self.fan_range, self.fan_memb[2], 'r', linewidth=0.5, linestyle='--')
            ax0.fill_between(self.fan_range, np.zeros_like(self.fan_range), 
                              self.aggregated, facecolor='Orange', alpha=0.7)
            ax0.plot([self.fan_defuzz, self.fan_defuzz], 
                      [0, fan_activation], 'k', linewidth=1.5, alpha=0.9)
            ax0.set_title('Aggregated membership and result (line)')
            
            # Turn off top/right axes
            for ax in (ax0,):
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.get_xaxis().tick_bottom()
                ax.get_yaxis().tick_left()

            plt.tight_layout()

        return self.fan_output        
    
    def plot_output_control(self, save=False):
        '''
        Plots the control map in 3D
        '''
        fan_output_range = []

        #Custom colormap
        colors = ['#179C7D', '#FFDC00', '#E2001A']  # FhG: B -> G -> R
        n_bins = 100  # Discretizes the interpolation into bins
        cmap_name = 'FhG_RGB'
        cm = LinearSegmentedColormap.from_list(
                cmap_name, colors, N=n_bins)
                        
        #Intepretation of membership functions
        for j in self.RH_range:
            for i in self.CO2_range:
                fan_output_range = np.append(fan_output_range,
                                              self.interpret_membership(j,i))
                
        #from matplotlib import cm
        xx, yy = np.meshgrid(self.CO2_range, self.RH_range)       
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        plt.subplots_adjust(bottom = 0.08, top = 0.96,
                        left = 0.04, right = 0.92) 
        ax.plot_surface(xx, yy*100, 
                        fan_output_range.reshape(len(self.RH_range),
                                                  len(self.CO2_range))*100, 
                          cmap=cm , rstride=1, cstride=1,
                          antialiased=True)
#        ax.zaxis.set_ticks(np.arange(0.2,0.9,0.1))
        ax.set_ylabel('Relative humidity [%]',fontsize = 13,labelpad=10)                
        ax.set_xlabel('$CO_2$ [ppm]',fontsize = 13,labelpad=10)                
        ax.set_zlabel('Fan speed [%]',fontsize = 13,labelpad=10)                
        ax.tick_params(labelsize = 10, axis='both')                
                        
        if save:
            fig.savefig('fuzzy_output.svg',format='svg',dpi=600)
            fig.savefig('fuzzy_output.png',format='png',dpi=300)
            fig.savefig('fuzzy_output.pdf',format='pdf',dpi=600)

#-------------------------------------------------------------------------------#            


class HeatingCurve:
    '''
    --------------------------------------------------------------
    Heating curve model for heat pump controlling. The inputs are the 
    nominal heat pump parameters, and the output is the temperature
    set point for the floor heating/radiator system. 
    The model is based on the heating curve model of the Modelica
    Buildings library (https://simulationresearch.lbl.gov/modelica/). 
    --------------------------------------------------------------              
    '''
    def __init__(self, **kwargs): 
        '''
        Initial conditions for the model
        Allowed keyword arguments (**kwargs):
    	----------	
        § T_room_set: float (default = 293.15)
            room temperature set point, in K
        § T_room_nom: float (default = 293.15)
            room nominal temperature set point, in K
        § T_sup_nom: float (default = 333.15)
            nominal heating system supply temperature set point, in K
        § T_ret_nom: float (default = 318.15)
            nominal heating system return temperature set point, in K
        § T_amb_nom: float (default = 261.15)
            nominal ambient temperature, in K
        § T_amb_lim: float (default = 293.15)
            nominal lowest ambient temperature, in K
        § heatingexp: float (default = 1.3)
            heating curve exponent, limited between 0 and 3, dimensionless            
        '''
               
        allowed_keys = {'T_room_set', 'T_room_nom', 'T_sup_nom', 'T_ret_nom',
                        'T_amb_nom','T_amb_lim', 'heatingexp'}

        rejected_keys = set(kwargs.keys()) - set(allowed_keys)
        if rejected_keys:
            raise ValueError("Invalid arguments in constructor: {}".
                             format(rejected_keys))
        else:
            self.default_values()
            self.__dict__.update((k, v) for k, v in kwargs.items() 
            if k in allowed_keys)
            self.default_curve_prop()
           
           
    def default_values(self):
        '''
        Reset the values to the default ones
        '''

        T_room_set = 293.15
        T_room_nom = 293.15
        T_sup_nom = 333.15
        T_ret_nom  = 318.15
        T_amb_nom  = 261.15
        T_amb_lim  = 293.15
        heatingexp = 1.3


    def default_curve_prop(self):
        '''
        Calculate the default properties of the heating curve
        '''

        self.heatingexp       = min(max(heatingexp, 0.0), 3.0)
        self.dTAmbHeaBal      = T_room_set - T_amb_lim
        self.T_amb_offset_nom = T_amb_nom + self.dTAmbHeaBal


    def curve_calc(self, T_amb):
        '''
        Calculate the system actual supply and return temperatures
        as a function of the ambient temperature. 
        Parameters:
    	----------	           
        § T_amb: float 
            ambient temperature, in K     
        Returns:
    	----------	           
        § T_sup: float 
            system supply temperature, in K            
        § T_ret: float 
            system return temperature, in K                
        '''
        self.T_amb = T_amb
        
        T_amb_offset = self.T_amb + self.dTAmbHeaBal
        
        # Relative heating load, compared to nominal conditions
        self.qRel = max(0., (self.T_room_set - T_amb_offset)/(self.T_room_nom - self.T_amb_offset_nom)) 
        
        # Determine supply and return temperatures
        T_sup = self.T_room_set + ((self.T_sup_nom + self.T_ret_nom)/2 - self.T_room_nom) * self.qRel**(1./self.heatingexp) + (self.T_sup_nom - self.T_ret_nom)/2. * self.qflow_rel
        T_ret = T_sup - self.qRel * (self.T_sup_nom - self.T_ret_nom)
        
        return [T_sup, T_ret]
              
        
    def update(self, T_amb, rr=False):
        '''
        Update the value of the heating curve according to the ambient temperature
        Parameters:
    	----------	           
        § T_amb: float 
            ambient temperature, in K     
        Returns:
    	----------	           
        § T_sup: float 
            system supply temperature, in K            
        § T_ret: float 
            system return temperature, in K     
        '''
        if np.isscalar(T_amb):
            [self.T_sup, self.T_ret] = self.curve_calc(T_amb)
            
        else:
            self.T_sup = np.zeros_like(T_amb)
            self.T_ret = np.zeros_like(T_amb)

            for i in range(len(T_amb)):
                [self.T_sup[i], self.T_ret[i]] = self.curve_calc(T_amb[i])

        if rr:
            return [self.T_sup, self.T_ret]

   
if __name__ == '__main__':
    pass

