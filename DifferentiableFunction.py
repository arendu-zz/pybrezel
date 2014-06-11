#! /usr/bin/env python

# Xuchen Yao, 9/23/2011, first draft
# Xuchen Yao, 9/26/2011, add L-BFGS support (CG is too slow)
# Xuchen Yao, 10/02/2011, major bug fix, add return status
import numpy as np
from numpy import asarray, zeros


class DifferentiableFunction:
    '''
    Wrapper class for:
    1. A multivariant differentiable function
    2. SciPy fmin_cg() method with nonlinear conjugate gradient algorithm.

    You should instantiate this class and provide its minimize()/maximize() function 
        with 2 other functions of your own:
    1. value(): your function definition.
    2. gradient(): the gradient of your function.

    References:
    http://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html
    http://www.scipy.org/doc/api_docs/SciPy.optimize.optimize.html#fmin_cg

    You need SciPy/NumPy to run this.
    '''

    def __init__(self, value_func, gradient_func, debug=False, method='LBFGS'):
        # "CG" for Conjugate Gradient
        # "LBFGS" for L-BFGS
        # In a small experiment I did, the CG method didn't converge even after 7,000+ iterations,
        # while "LBFGS" method converges within 20 iterations, thus defaulting to "LBFGS" method
        self.method = method
        self.value_func, self.gradient_func = value_func, gradient_func
        self.size = None
        self.feature2index = {}
        self.index2feature = {}
        # SciPy by default only optimizes minimal objective. To get maximum,
        # we revert all value() and gradient() function values.
        self.revert = False
        self.debug = debug
        if self.debug:
            self.h = 10 ** -14
            self.h_times_gradient = None
            self.f_theta_h_debug = None
        return

    def fprime(self, theta):
        self.size = len(theta)
        initials = []
        index = 0
        self.feature2index, self.index2feature = {}, {}
        for feature, value in theta.items():
            self.feature2index[feature] = index
            self.index2feature[index] = feature
            initials.append(float(value))
            index += 1
        from scipy.optimize import approx_fprime

        eps = np.sqrt(np.finfo(np.float).eps)
        return approx_fprime(initials, self.value_translator, [eps] * self.size)


    def finite_diff(self, theta):
        self.size = len(theta)
        initials = []
        index = 0
        self.feature2index, self.index2feature = {}, {}
        for feature, value in theta.items():
            self.feature2index[feature] = index
            self.index2feature[index] = feature
            initials.append(float(value))
            index += 1
        from scipy.optimize import check_grad

        return check_grad(self.value_translator, self.gradient_translator, initials)


    def value_translator(self, point):
        '''We are trying to optimize a function of a parameter vector.
           SciPy assumes that this vector will be a list [] that maps parameter 
           NUMBERS to parameter values.
           The user's value() function assumes that this vector will be a 
           map {} that maps parameters NAMES to parameter values.
           Here we provide a value function of the sort SciPy wants,
           by wrapping around the user's value function.'''
        theta = {}
        for i in range(len(point)):
            theta[self.index2feature[i]] = point[i]
        if self.debug:
            theta_h_debug = {}
            for i in range(len(point)):
                theta_h_debug[self.index2feature[i]] = point[i] + self.h
        if self.revert:
            ret = -self.value_func(theta)
            if self.debug:
                self.f_theta_h_debug = -self.value_func(theta_h_debug)
                print "f(theta+h): %.14f" % self.f_theta_h_debug
                f_theta_plus_h_gradient = self.h_times_gradient + ret
                print "f(theta) + h . gradient(theta): %.14f" % f_theta_plus_h_gradient
                print "difference: %.14f" % (self.f_theta_h_debug - f_theta_plus_h_gradient)
            return ret
        else:
            ret = self.value_func(theta)
            if self.debug:
                self.f_theta_h_debug = self.value_func(theta_h_debug)
                print "f(theta+h): %.14f" % self.f_theta_h_debug
                f_theta_plus_h_gradient = self.h_times_gradient + ret
                print "f(theta) + h . gradient(theta): %.14f" % f_theta_plus_h_gradient
                print "difference: %.14f" % (self.f_theta_h_debug - f_theta_plus_h_gradient)
            return ret

    def gradient_translator(self, point):
        '''Similar to value_translator(), wraps around the user's gradient() function.'''
        theta = {}
        for i in range(len(point)):
            theta[self.index2feature[i]] = point[i]
        theta = self.gradient_func(theta)
        if self.debug:
            self.h_times_gradient = self.h * sum(theta.values())
            # in case you changed the size of your feature function
        if self.size != len(theta):
            raise Exception("Size of your feature vector (%d) doesn't match size of the gradients (%d)!" % (self.size, len(theta)))
        gradients = zeros(len(theta))
        for feature, value in theta.items():
            if self.revert:
                value = -value
            gradients[self.feature2index[feature]] = value
        return gradients

    def minimize(self, theta):
        '''Get the minimal value of this differentiable function
           input: theta with initial values, after function return, theta will be set with optimal values
           output: the minimal value '''
        return self.optimize(theta)

    def maximize(self, theta):
        '''Get the maximal value of this differentiable function
           input: theta with initial values, after function return, theta will be set with optimal values
           output: the maximal value '''
        (fopt, theta, return_status) = self.optimize(theta, revert=True)
        return (-fopt, theta, return_status)

    def optimize(self, theta, revert=False):
        '''real executing function'''
        self.size = len(theta)
        self.revert = revert
        initials = []
        index = 0
        self.feature2index, self.index2feature = {}, {}
        for feature, value in theta.items():
            self.feature2index[feature] = index
            self.index2feature[index] = feature
            initials.append(float(value))
            index += 1
        if self.method == "LBFGS":
            from scipy.optimize import fmin_l_bfgs_b
            # http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
            (xopt, fopt, return_status) = fmin_l_bfgs_b(self.value_translator, initials, self.gradient_translator, pgtol=0.01)
            #print "============Optimization by LBFGS returns: ", return_status['task']
        elif self.method == "CG":
            from scipy.optimize import fmin_cg
            # http://www.scipy.org/doc/api_docs/SciPy.optimize.optimize.html#fmin_cg
            (xopt, fopt, _, _, return_status) = fmin_cg(self.value_translator, initials, self.gradient_translator, full_output=1, disp=0)
            #print "============CG: ", return_status
        else:
            raise Exception("No optimization method defined!")
        self.size = None
        for i, x in enumerate(xopt):
            theta[self.index2feature[i]] = x
        return (fopt, theta, return_status)

    def iterationCallback(self, current_theta):
        print "iteration complete"

