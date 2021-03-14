
__author__ = 'Pedro Pablo'


from Equation import Expression


class IVP:
    """
    Contains the formulation of the IVP to be solved. 

    `function` = right part of the ODE to be solved.
    `x0` = initial point.
    `y0` = value of the solution at point `x0`.
    """
    def __init__(self, function, x0, y0):
        self.func = function
        self.x_0 = x0
        self.y_0 = y0

    def convert_to_keras_expression(self):
        d = {'sin': 'K.sin', 'sen': 'K.sin', 'cos': 'K.cos', 'tan': 'K.tan',
             'log': 'K.log', 'ln': 'K.log', 'exp': 'K.exp',
             'sinh': 'K.sinh', 'senh': 'K.sinh', 'cosh': 'K.cosh',
             'tanh': 'K.tanh', 'sqrt': 'K.sqrt'}
        for key in d.keys():
            self.func = self.func.replace(key, d[key])
        return self.func

    def function(self):
        return Expression(self.func, ["x", "y"])

    def x0(self):
        return self.x_0

    def y0(self):
        return self.y_0
