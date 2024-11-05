# Կազմել ծրագիր, որը գրադիենտի պրոյեկտման մեթոդով կիրականացնի f(x)=1/2(Ax, x) + (b, x) 
# քառակուսային ֆունկցիայի մինիմիզացիան 1-ին քառորդում, շրջանագծի վրա և ուղանկյունանիստի վրա։ 
# L հաստատունը վերցնել A մատրիցի նորմը, ուժեղ ուռուցիկության teta հաստատունը վերցնել 
# մատրիցի մինիմալ սեթական արժեքը, իսկ ak=2*teta/L^2 : Կանգառի քայլ համարել 
# ||x^(k+1) - x^k||<epsilon0 պայմանը, որտեղ epsilon0>0 նախապես տրված ճշտություն է։

import numpy as np

def quadratic_function(A, b, x):
    """ Calculate the quadratic function f(x) = 1/2 * (Ax, x) + (b, x) """
    return 0.5 * np.dot(x.T, np.dot(A, x)) + np.dot(b, x)

def gradient(A, b, x):
    """ Calculate the gradient of f(x) """
    return np.dot(A, x) + b

def project_to_first_quadrant(x):
    """ Project x onto the 1st quadrant (x1 >= 0, x2 >= 0) """
    return np.maximum(x, 0)

def project_to_circle(x, radius):
    """ Project x onto the circle of given radius """
    norm = np.linalg.norm(x)
    if norm > radius:
        return (radius / norm) * x
    return x

def project_to_rectangle(x, rect_min, rect_max):
    """ Project x onto the rectangle defined by rect_min and rect_max """
    return np.clip(x, rect_min, rect_max)

def gradient_projection_method(A, b, x0, radius=None, rect_min=None, rect_max=None, epsilon=1e-6, max_iter=1000):
    """ Gradient projection method for minimizing the quadratic function """
    L = np.linalg.norm(A)  # Norm of the matrix A
    theta = np.min(np.linalg.eigvals(A))  # Minimum eigenvalue for strong convexity
    ak = 2 * theta / (L ** 2)  # Step size
    
    x = x0
    for k in range(max_iter):
        grad = gradient(A, b, x)
        x_new = x - ak * grad  # Gradient descent step

        # Project onto the feasible region based on specified constraints
        if radius is not None:
            x_new = project_to_circle(x_new, radius)
        if rect_min is not None and rect_max is not None:
            x_new = project_to_rectangle(x_new, rect_min, rect_max)
        x_new = project_to_first_quadrant(x_new)

        # Stopping condition
        if np.linalg.norm(x_new - x) < epsilon:
            break

        x = x_new

    return x, quadratic_function(A, b, x)

# Example usage:
A = np.array([[2, 0], [0, 2]])  # Example positive definite matrix A
b = np.array([-2, -5])           # Example linear term b
x0 = np.array([0.5, 0.5])        # Initial guess
radius = 1.0                     # Circle radius for projection
rect_min = np.array([0, 0])      # Rectangle minimum
rect_max = np.array([1, 1])      # Rectangle maximum

# Call the optimization function
optimal_x, optimal_value = gradient_projection_method(A, b, x0, radius, rect_min, rect_max)

print("Optimal x:", optimal_x)
print("Optimal value of f(x):", optimal_value)
