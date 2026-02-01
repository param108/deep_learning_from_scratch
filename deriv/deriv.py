from typing import Callable, List
from numpy import ndarray
import numpy as np
import matplotlib.pyplot as plt

Chain= List[Callable[[ndarray], ndarray]] 

def chain_length_2(chain: Chain, x: ndarray) -> ndarray:
    """
    Compute the result of applying a chain of two functions to an input.

    Parameters:
    chain : Chain
        A list containing two functions to be applied in sequence.
    x : ndarray
        The input array to which the functions will be applied.

    Returns:
    ndarray
        The result of applying the two functions in the chain to the input x.
    """
    return chain[1](chain[0](x))

def chain_length_n(chain: Chain, x: ndarray) -> ndarray:
    """
    Compute the result of applying a chain of n functions to an input.

    Parameters:
    chain : Chain
        A list of functions to be applied in sequence.
    x : ndarray
        The input array to which the functions will be applied.

    Returns:
    ndarray
        The result of applying all functions in the chain to the input x.
    """
    result = x
    for func in chain:
        result = func(result)
    return result

def deriv_chain_n(chain: Chain, x: ndarray, h: float = 1e-5) -> ndarray:
    """
    Compute the numerical derivative of a chain of n functions at a given point.

    Parameters:
    chain : Chain
        A list of functions to be applied in sequence.
    x : ndarray
        The point at which to compute the derivative.
    h : float, optional
        The step size for the finite difference approximation (default is 1e-5).

    Returns:
    ndarray
        The numerical derivative of the chain of functions at point x.
    """
    derivatives = []
    current_input = x

    for func in chain:
        deriv_func = deriv(func, current_input, h)
        derivatives.append(deriv_func)
        current_input = func(current_input)

    total_derivative = np.ones_like(x)
    for deriv_value in derivatives:
        total_derivative *= deriv_value

    return total_derivative

def deriv(func: Callable[[ndarray], ndarray], x: ndarray, h: float = 1e-5) -> ndarray:
    """
    Compute the numerical derivative of a function at a given point using central difference.

    Parameters:
    func : Callable[[ndarray], ndarray]
        The function for which to compute the derivative.
    x : ndarray
        The point at which to compute the derivative.
    h : float, optional
        The step size for the finite difference approximation (default is 1e-5).

    Returns:
    ndarray
        The numerical derivative of the function at point x.
    """
    return (func(x + h) - func(x - h)) / (2 * h)

def deriv_chain_2(chain: Chain, x: ndarray, h: float = 1e-5) -> ndarray:
    """
    Compute the numerical derivative of a chain of functions at a given point.

    Parameters:
    chain : Chain
        A list of functions to be applied in sequence.
    x : ndarray
        The point at which to compute the derivative.
    h : float, optional
        The step size for the finite difference approximation (default is 1e-5).

    Returns:
    ndarray
        The numerical derivative of the chain of functions at point x.
    """
    func0 = chain[0]
    func1 = chain[1]

    deriv_func1 = deriv(func1, func0(x), h)
    deriv_func0 = deriv(func0, x, h)

    return deriv_func1 * deriv_func0

def deriv_chain_3(chain: Chain, x: ndarray, h: float = 1e-5) -> ndarray:
    """
    Compute the numerical derivative of a chain of three functions at a given point.

    Parameters:
    chain : Chain
        A list of three functions to be applied in sequence.
    x : ndarray
        The point at which to compute the derivative.
    h : float, optional
        The step size for the finite difference approximation (default is 1e-5).

    Returns:
    ndarray
        The numerical derivative of the chain of functions at point x.
    """
    func0 = chain[0]
    func1 = chain[1]
    func2 = chain[2]

    deriv_func2 = deriv(func2, func1(func0(x)), h)
    deriv_func1 = deriv(func1, func0(x), h)
    deriv_func0 = deriv(func0, x, h)

    return deriv_func2 * deriv_func1 * deriv_func0
def sigmoid(x: ndarray) -> ndarray:
    """
    Compute the sigmoid activation function.

    Parameters:
    x : ndarray
        The input array.

    Returns:
    ndarray
        The result of applying the sigmoid function to the input.
    """
    return 1 / (1 + np.exp(-(x)))

def square(x: ndarray) -> ndarray:
    """
    Compute the square of the input.

    Parameters:
    x : ndarray
        The input array.

    Returns:
    ndarray
        The square of the input array.
    """
    return np.power(x, 2)

def leaky_relu(x: ndarray, alpha: float = 0.2) -> ndarray:
    """
    Compute the Leaky ReLU activation function.

    Parameters:
    x : ndarray
        The input array.
    alpha : float, optional
        The slope for negative input values (default is 0.2).

    Returns:
    ndarray
        The result of applying the Leaky ReLU function to the input.
    """
    return np.where(x > 0, x, alpha * x)

def plot_chain_2(chain: Chain, plot_range: ndarray) -> None:
    """
    Plot the functions in the chain and their derivatives.

    Parameters:
    chain : Chain
        A list of functions to be plotted.
    plot_range : ndarray
        The range of input values for plotting.
    """

    y_values = chain_length_2(chain, plot_range)
    
    print(y_values)

    dy_values = deriv_chain_2(chain, plot_range)

    # Compute individual function outputs
    func0_output = chain[0](plot_range)
    func1_output = chain[1](plot_range)
    
    plt.figure(figsize=(18, 6))

    # Plot first function
    plt.subplot(1, 4, 1)
    plt.plot(plot_range, func0_output, label='f1(x)', color='blue')
    plt.title('First Function (f1)')
    plt.xlabel('Input')
    plt.ylabel('f1(x)')
    plt.grid()
    plt.legend()

    # Plot second function (with transformed input)
    plt.subplot(1, 4, 2)
    plt.plot(plot_range, func1_output, label='f2(x)', color='green')
    plt.title('Second Function (f2)')
    plt.xlabel('Input')
    plt.ylabel('f2(x)')
    plt.grid()
    plt.legend()

    # Plot chain output
    plt.subplot(1, 4, 3)
    plt.plot(plot_range, y_values, label='Chain Output', color='purple')
    plt.title('Output of Function Chain')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.grid()
    plt.legend()

    # Plot derivative of chain
    plt.subplot(1, 4, 4)
    plt.plot(plot_range, dy_values, label='Derivative of Chain', color='orange')
    plt.title('Derivative of Function Chain')
    plt.xlabel('Input')
    plt.ylabel('Derivative')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

def test_deriv_chain_2() -> None:
    chain = [square, sigmoid]
    PLOT_RANGE = np.arange(-3, 3, 0.1)
    print(square(PLOT_RANGE))
    plot_chain_2(chain, PLOT_RANGE)

def plot_chain_n(chain: Chain, plot_range: ndarray) -> None:
    """
    Plot the functions in the chain and their derivatives for n functions.

    Parameters:
    chain : Chain
        A list of functions to be plotted.
    plot_range : ndarray
        The range of input values for plotting.
    """

    y_values = chain_length_n(chain, plot_range)
    dy_values = deriv_chain_n(chain, plot_range)

    plt.figure(figsize=(12, 6))

    # Plot chain output
    plt.subplot(1, 2, 1)
    plt.plot(plot_range, y_values, label='Chain Output', color='purple')
    plt.title('Output of Function Chain')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.grid()
    plt.legend()

    # Plot derivative of chain
    plt.subplot(1, 2, 2)
    plt.plot(plot_range, dy_values, label='Derivative of Chain', color='orange')
    plt.title('Derivative of Function Chain')
    plt.xlabel('Input')
    plt.ylabel('Derivative')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

def test_deriv_chain_n() -> None:
    chain = [leaky_relu, sigmoid, square]
    PLOT_RANGE = np.arange(-3, 3, 0.1)
    plot_chain_n(chain, PLOT_RANGE)

test_deriv_chain_n()