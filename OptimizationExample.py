# Xuchen Yao, 9/23/2011, first draft
# Xuchen Yao, 9/26/2011, add L-BFGS support (CG is too slow)
# Xuchen Yao, 10/02/2011, major bug fix, add return status

'''
Sample code for using a Conjugate Gradient Optimizer to
find out a maximal/minimal value of a linear/nonlinear function.

You need SciPy/NumPy to run this.
'''

import sys

sys.path.append(".")
sys.path.append("/usr/local/data/cs465/hw-lm/code/python")
from DifferentiableFunction import DifferentiableFunction


# function definition: f(x,y) = -(x-100)^2+5xy-(y-200)^4
# 
#    Here the arguments x and y are actually passed to the function as
#    elements of theta, which packages up all of the named arguments.
#    In log-linear modeling, you will similarly use theta to package
#    up the weights of all of the named features.
def value(theta):
    print 'recived theta', theta
    return -(theta["x"] - 100) ** 2 + 5 * theta["x"] * theta["y"] - (theta["y"] - 200) ** 4


# the gradient of the function defined in value() above
def gradient(theta):
    gradient_map = {}
    gradient_map["x"] = -2 * (theta["x"] - 100) + 5 * theta["y"]
    gradient_map["y"] = 5 * theta["x"] - 4 * (theta["y"] - 200) ** 3
    print 'returning gradient', gradient_map
    return gradient_map

# theta is a feature vector with initialized values
theta = {"x": 0.0, "y": 0.0}
F = DifferentiableFunction(value, gradient)
#fd = F.fprime(theta)
#print 'finite diff', fd
(fopt, theta, return_status) = F.maximize(theta)
#after optimization, theta now has the optimal feature values
print "Max f(x) = %f at x = %f y = %f" % (fopt, theta.get("x"), theta.get("y"))
# always check the return status. Sometimes the optimizer returns quickly but it doesn't converge!
print return_status