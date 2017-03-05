from numbers import Real
from collections import OrderedDict
from numpy import reshape, prod, ndarray
from sympy import Expr, Symbol, lambdify
from typing import Sequence, Dict, Callable, Tuple
import numpy as np


# class ExpressionVector:
#     def __init__(self, expression: Sequence[Expr], parameters: 'OrderedDict[str, Sequence[Symbol]]'):
#         self.expression = np.array(expression)
#         self.parameters = parameters
#         self.expression_cache = dict()  # type: Dict[Tuple[str], np.array[Expr]]
#         self.function_cache = dict()  # type: Dict[Tuple[str], Callable]
#
#     def to_function(self, derivatives: Tuple[str] = ()):
#         function = self.function_cache.get(derivatives)
#         if function is None:
#             # Search for the derivative that is furthest along
#             derivatives_covered = 0
#             expression_array = self.expression
#             for i in range(1, len(derivatives)+1):
#                 temp = self.expression_cache.get(derivatives[0:i])
#                 if temp is not None:
#                     derivatives_covered = i
#                     expression_array = temp
#                 else:
#                     break
#
#             # Compute the rest of the derivatives
#             for i in range(derivatives_covered+1, len(derivatives)):
#                 derivative_parameter = derivatives[i]
#                 derivative_symbolics = self.parameters[derivative_parameter]
#                 expression_array = multidimensional_derivative(expression_array, derivative_symbolics)
#
#                 # Cache the derivative
#                 derivative_key = derivatives[0:i]
#                 self.expression_cache[derivative_key] = expression_array
#
#             # Convert expressions to multidimensional function
#
#
#             # Cache the computed value
#             self.function_cache[derivatives] = function
#
#         return function


def multidimensional_derivative(expressions: np.ndarray, symbols: Sequence['Symbol|str']):
    if ~isinstance(expressions, np.ndarray):
        expressions = np.array(expressions, dtype=Expr)

    dimensions = expressions.shape + (len(symbols),)
    derivative = np.empty(dimensions, dtype=Expr)

    for i_expression in np.ndindex(expressions.shape):
        for i_symbol in range(len(symbols)):
            index = i_expression + (i_symbol,)
            expression = expressions[i_expression]
            if isinstance(expression, Real):
                derivative[index] = 0
            else:
                derivative[index] = expression.diff(symbols[i_symbol])

    return derivative


def multidimensional_lambdify(variables, expression: ndarray):
    shape = expression.shape
    n = prod(shape)
    expression = tuple(expression.reshape((n,)))  # tuple required because lambdify does not work on ndarray

    function = lambdify(variables, expression)

    return lambda *values: reshape(function(*values), shape)
