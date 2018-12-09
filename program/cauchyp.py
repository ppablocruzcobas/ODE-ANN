# -*- coding: utf-8 -*-
__author__ = "Pedro Pablo"


class IVP:
    """
    Contains the formulation of the IVP to be solved and once the network is
    trained the solution at any point within the interval may be obtained just
    calling the instance at desire time (can be a list).

    `eqfunc` = right part of the ODE to be solved.
    `t0` = initial time.
    `x0` = value of the solution at time `t0`.
    """
    def __init__(self, eqfunc, t0, x0):
        self.__eqfunc = eqfunc
        self.__t0 = t0
        self.__x0 = x0
        self.__trial_solution = None

    def __call__(self, t):
        if self.__trial_solution is None:
            raise Exception("""You must first solve the IVP calling solve_ivp
            in the neural net and passing this object as an argument""")
        else:
            return self.__trial_solution(t, True)

    def set_solution(self, trial_solution):
        """
        To be called only from inside the neural net once
        the training is complete.
        """
        self.__trial_solution = trial_solution

    def get_equation(self):
        return self.__eqfunc

    def get_t0(self):
        return self.__t0

    def get_x0(self):
        return self.__x0
