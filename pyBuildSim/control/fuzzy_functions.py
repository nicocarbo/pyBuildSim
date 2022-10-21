# -*- coding: utf-8 -*-

"""
author: Nicolas Carbonare
mail: nicocarbonare@gmail
last updated: June 06, 2021
"""

import numpy as np

class fuzzy_func:
    '''
    Fuzzy functions - taken from package scikit-fuzzy 
    
    https://github.com/scikit-fuzzy/scikit-fuzzy
    '''
    def __init__(self):
        '''
        Blank initialization
        ''' 
  
    def defuzz_centroid(self, x, mfx):
        '''
        Defuzzification using centroid (`center of gravity`) method.
        Parameters
        ----------
        x : 1d array, length M
            Independent variable
        mfx : 1d array, length M
            Fuzzy membership function
        Returns
        -------
        u : 1d array, length M
            Defuzzified result
        '''
    
        '''
        As we suppose linearity between each pair of points of x, we can calculate
        the exact area of the figure (a triangle or a rectangle).
        '''

        sum_moment_area = 0.0
        sum_area = 0.0

        # If the membership function is a singleton fuzzy set:
        if len(x) == 1:
            return x[0]*mfx[0] / np.fmax(mfx[0], np.finfo(float).eps).astype(float)
    
        # else return the sum of moment*area/sum of area
        for i in range(1, len(x)):
            x1 = x[i - 1]
            x2 = x[i]
            y1 = mfx[i - 1]
            y2 = mfx[i]
    
            # if y1 == y2 == 0.0 or x1==x2: --> rectangle of zero height or width
            if not(y1 == y2 == 0.0 or x1 == x2):
                if y1 == y2:  # rectangle
                    moment = 0.5 * (x1 + x2)
                    area = (x2 - x1) * y1
                elif y1 == 0.0 and y2 != 0.0:  # triangle, height y2
                    moment = 2.0 / 3.0 * (x2-x1) + x1
                    area = 0.5 * (x2 - x1) * y2
                elif y2 == 0.0 and y1 != 0.0:  # triangle, height y1
                    moment = 1.0 / 3.0 * (x2 - x1) + x1
                    area = 0.5 * (x2 - x1) * y1
                else:
                    moment = (2.0 / 3.0 * (x2-x1) * (y2 + 0.5*y1)) / (y1+y2) + x1
                    area = 0.5 * (x2 - x1) * (y1 + y2)
    
                sum_moment_area += moment * area
                sum_area += area

        return sum_moment_area / np.fmax(sum_area,
                                         np.finfo(float).eps).astype(float)
    
    
    def trapmf(self,x, abcd):
        '''
        Trapezoidal membership function generator.
        Parameters
        ----------
        x : 1d array
            Independent variable.
        abcd : 1d array, length 4
            Four-element vector.  Ensure a <= b <= c <= d.
        Returns
        -------
        y : 1d array
            Trapezoidal membership function.
        '''
        assert len(abcd) == 4, 'abcd parameter must have exactly four elements.'
        a, b, c, d = np.r_[abcd]
        assert a <= b and b <= c and c <= d, 'abcd requires the four elements \
                                              a <= b <= c <= d.'
        y = np.ones(len(x))
    
        idx = np.nonzero(x <= b)[0]
        y[idx] = self.trimf(x[idx], np.r_[a, b, b])
    
        idx = np.nonzero(x >= c)[0]
        y[idx] = self.trimf(x[idx], np.r_[c, c, d])
    
        idx = np.nonzero(x < a)[0]
        y[idx] = np.zeros(len(idx))
    
        idx = np.nonzero(x > d)[0]
        y[idx] = np.zeros(len(idx))
    
        return y
    
    
    def trimf(self,x, abc):
        '''
        Triangular membership function generator.
        Parameters
        ----------
        x : 1d array
            Independent variable.
        abc : 1d array, length 3
            Three-element vector controlling shape of triangular function.
            Requires a <= b <= c.
        Returns
        -------
        y : 1d array
            Triangular membership function.
        '''
        assert len(abc) == 3, 'abc parameter must have exactly three elements.'
        a, b, c = np.r_[abc]     # Zero-indexing in Python
        assert a <= b and b <= c, 'abc requires the three elements a <= b <= c.'
    
        y = np.zeros(len(x))
    
        # Left side
        if a != b:
            idx = np.nonzero(np.logical_and(a < x, x < b))[0]
            y[idx] = (x[idx] - a) / float(b - a)
    
        # Right side
        if b != c:
            idx = np.nonzero(np.logical_and(b < x, x < c))[0]
            y[idx] = (c - x[idx]) / float(c - b)
    
        idx = np.nonzero(x == b)
        y[idx] = 1
        return y
    
    def sigmf(self, x, b, c):
        """
        The basic sigmoid membership function generator.
        Parameters
        ----------
        x : 1d array
            Data vector for independent variable.
        b : float
            Offset or bias.  This is the center value of the sigmoid, where it
            equals 1/2.
        c : float
            Controls 'width' of the sigmoidal region about `b` (magnitude); also
            which side of the function is open (sign). A positive value of `a`
            means the left side approaches 0.0 while the right side approaches 1.;
            a negative value of `c` means the opposite.
        Returns
        -------
        y : 1d array
            Generated sigmoid values, defined as y = 1 / (1. + exp[- c * (x - b)])
        Notes
        -----
        These are the same values, provided separately and in the opposite order
        compared to the publicly available MathWorks' Fuzzy Logic Toolbox
        documentation. Pay close attention to above docstring!
        """
        return 1. / (1. + np.exp(- c * (x - b)))
    
    def gaussmf(self, x, mean, sigma):
        """
        Gaussian fuzzy membership function.
        Parameters
        ----------
        x : 1d array or iterable
            Independent variable.
        mean : float
            Gaussian parameter for center (mean) value.
        sigma : float
            Gaussian parameter for standard deviation.
        Returns
        -------
        y : 1d array
            Gaussian membership function for x.
        """
        return np.exp(-((x - mean)**2.) / (2 * sigma**2.))
    
    def interp_membership(self, x, xmf, xx):
        """
        Find the degree of membership ``u(xx)`` for a given value of ``x = xx``.
    
        Parameters
        ----------
        x : 1d array
            Independent discrete variable vector.
        xmf : 1d array
            Fuzzy membership function for ``x``.  Same length as ``x``.
        xx : float
            Discrete singleton value on universe ``x``.
    
        Returns
        -------
        xxmf : float
            Membership function value at ``xx``, ``u(xx)``.
    
        Notes
        -----
        For use in Fuzzy Logic, where an interpolated discrete membership function
        u(x) for discrete values of x on the universe of ``x`` is given. Then,
        consider a new value x = xx, which does not correspond to any discrete
        values of ``x``. This function computes the membership value ``u(xx)``
        corresponding to the value ``xx`` using linear interpolation.  
        """
        # Nearest discrete x-values
        x1 = x[x <= xx][-1]
        x2 = x[x >= xx][0]
    
        idx1 = np.nonzero(x == x1)[0][0]
        idx2 = np.nonzero(x == x2)[0][0]
    
        xmf1 = xmf[idx1]
        xmf2 = xmf[idx2]
    
        if x1 == x2:
            xxmf = xmf[idx1]
        else:
            slope = (xmf2 - xmf1) / float(x2 - x1)
            xxmf = slope * (xx - x1) + xmf1
    
        return xxmf
        
        
if __name__ == '__main__':
    pass
