# 15 125
# Կազմել ծրագիր, որը գրադիենտի պրոյեկտման մեթոդով կիրականացնի f(x)=1/2(Ax, x) + (b, x) 
# քառակուսային ֆունկցիայի մինիմիզացիան շրջանագծի վրա, ուղղանկյան վրա և ուղանկյունանիստի վրա։ 
# L հաստատունը վերցնել A մատրիցի նորմը, ուժեղ ուռուցիկության teta հաստատունը վերցնել 
# մատրիցի մինիմալ սեփական արժեքը, իսկ ak=2*teta/L^2 : Կանգառի քայլ համարել 
# ||x^(k+1) - x^k||<epsilon0 պայմանը, որտեղ epsilon0>0 նախապես տրված ճշտություն է։

import numpy as np

def quadratic_function(A, b, x):
    # f(x) = 1/2 * (Ax, x) + (b, x)
    return 0.5 * np.dot(x.T, np.dot(A, x)) + np.dot(b, x)

def gradient(A, b, x):
    # f(x)
    return np.dot(A, x) + b

# Եթե «x»-ի նորմը գերազանցում է «շառավիղը», ապա այն «x»-ի մասշտաբով նվազում է, 
# ուստի այն գտնվում է շրջանագծի սահմանի վրա; հակառակ դեպքում, այն վերադարձնում է `x` անփոփոխ:
def project_to_circle(x, radius):
    # Project x onto the circle of given radius
    norm = np.linalg.norm(x)
    if norm > radius:
        return (radius / norm) * x
    return x

def project_to_rectangle(x, rect_min, rect_max):
    # Project x onto the rectangle defined by rect_min and rect_max
    return np.clip(x, rect_min, rect_max)

def project_to_3d_rectangular_prism(x, x_min, x_max, y_min, y_max, z_min, z_max):
    # Clip each dimension independently
    x_proj = np.clip(x[0], x_min, x_max)
    y_proj = np.clip(x[1], y_min, y_max)
    z_proj = np.clip(x[2], z_min, z_max)
    
    # Return the projected point as a numpy array
    return np.array([x_proj, y_proj, z_proj])

def gradient_projection_method(A, b, x0, radius=None, rect_min=None, rect_max=None, prism_bounds=None, epsilon=1e-6, max_iter=1000):
    # Gradient projection method for minimizing the quadratic function
    L = np.linalg.norm(A, ord=2)  # 2-norm (spectral norm) of matrix A
    theta = np.min(np.linalg.eigvals(A))  # A matrixi minimal sepakan arjeq
    ak = 2 * theta / (L ** 2)  # Step size
    
    # Initial x = x0
    x = x0
    for k in range(max_iter):
        grad = gradient(A, b, x)
        # Gradient descent step: x_new = x^(k+1)
        x_new = x - ak * grad  

        # Apply projection based on the specified constraints (circle, rectangle, or 3D prism)
        if radius is not None:
            x_new = project_to_circle(x_new, radius)
        if rect_min is not None and rect_max is not None:
            if len(x) == 2:
                x_new = project_to_rectangle(x_new, rect_min, rect_max)
        if prism_bounds is not None and len(x) == 3:
            x_new = project_to_3d_rectangular_prism(
                x_new, prism_bounds['x_min'], prism_bounds['x_max'],
                prism_bounds['y_min'], prism_bounds['y_max'],
                prism_bounds['z_min'], prism_bounds['z_max']
            )
        
        # Stopping condition
        if np.linalg.norm(x_new - x) < epsilon:
            break

        x = x_new
    # Return the optimal x and the function value at x
    return x, quadratic_function(A, b, x)

def is_positive_semidefinite(matrix):
    """Check if a matrix is symmetric positive semidefinite."""
    return np.allclose(matrix, matrix.T) and np.all(np.linalg.eigvals(matrix) >= 0)

def get_matrix(prompt):
    """Prompt user for matrix input."""
    while True:
        try:
            rows = int(input(f"Enter the number of rows for {prompt}: "))
            print(f"Enter the matrix {prompt} row by row, values separated by spaces:")
            matrix = []
            for _ in range(rows):
                row = list(map(float, input().split()))
                matrix.append(row)
            matrix = np.array(matrix)
            if is_positive_semidefinite(matrix):
                return matrix
            else:
                print(f"{prompt} must be symmetric positive semidefinite. Try again.")
        except Exception as e:
            print(f"Invalid input for {prompt}: {e}. Try again.")

def get_vector(prompt, size):
    """Prompt user for vector input."""
    while True:
        try:
            print(f"Enter the {prompt} as {size} space-separated values:")
            vector = np.array(list(map(float, input().split())))
            if len(vector) == size:
                return vector
            else:
                print(f"Expected {size} values. Try again.")
        except Exception as e:
            print(f"Invalid input for {prompt}: {e}. Try again.")

def get_positive_scalar(prompt):
    """Prompt user for a positive scalar."""
    while True:
        try:
            scalar = float(input(f"Enter a positive value for {prompt}: "))
            if scalar > 0:
                return scalar
            else:
                print(f"{prompt} must be positive. Try again.")
        except Exception as e:
            print(f"Invalid input for {prompt}: {e}. Try again.")

def get_bounds(prompt, dimensions):
    """Prompt user for rectangle or prism bounds."""
    bounds = {}
    for dim in range(dimensions):
        while True:
            try:
                lower = float(input(f"Enter the minimum value for {prompt} dimension {dim + 1}: "))
                upper = float(input(f"Enter the maximum value for {prompt} dimension {dim + 1}: "))
                if lower <= upper:
                    bounds[f"dim_{dim}_min"] = lower
                    bounds[f"dim_{dim}_max"] = upper
                    break
                else:
                    print("Minimum must not exceed maximum. Try again.")
            except Exception as e:
                print(f"Invalid input for {prompt} dimension {dim + 1}: {e}. Try again.")
    return bounds

def gradient_projection_method(A, b, x0, r=None, rect_min=None, rect_max=None, prism_bounds=None):
    # Placeholder for optimization logic
    optimal_x = x0  # Example: use initial guess as placeholder
    optimal_value = np.dot(x0.T, np.dot(A, x0)) + np.dot(b, x0)  # Quadratic objective function
    return optimal_x, optimal_value

if __name__ == "__main__":
    # 2D case
    print("2D Optimization Case:")
    A = get_matrix("2D matrix A")
    b = get_vector("2D vector b", A.shape[0])
    x0 = get_vector("2D initial guess x0", A.shape[0])
    r = get_positive_scalar("radius r")
    rect_min = get_vector("rectangle min bounds", A.shape[0])
    rect_max = get_vector("rectangle max bounds", A.shape[0])

    optimal_x_radius, optimal_value_radius = gradient_projection_method(A, b, x0, r=r)
    print("Optimal x (radius):", optimal_x_radius)
    print("Optimal value of f(x) (radius):", optimal_value_radius)

    optimal_x, optimal_value = gradient_projection_method(A, b, x0, rect_min=rect_min, rect_max=rect_max)
    print("Optimal x (2D):", optimal_x)
    print("Optimal value of f(x) (2D):", optimal_value)

    # 3D case
    print("3D Optimization Case:")
    A_3d = get_matrix("3D matrix A")
    b_3d = get_vector("3D vector b", A_3d.shape[0])
    x0_3d = get_vector("3D initial guess x0", A_3d.shape[0])
    # prism_bounds = get_bounds("3D prism bounds", A_3d.shape[0])
    prism_bounds = {
        'x_min': 0, 'x_max': 1,
        'y_min': 0, 'y_max': 1,
        'z_min': 0, 'z_max': 1
    }

    optimal_x_3d, optimal_value_3d = gradient_projection_method(A_3d, b_3d, x0_3d, prism_bounds=prism_bounds)
    print("Optimal x (3D):", optimal_x_3d)
    print("Optimal value of f(x) (3D):", optimal_value_3d)


#     A = np.array([[2, 0], [0, 2]])  # Example positive definite matrix A
#     b = np.array([-2, -5])           # Example linear term b
#     x0 = np.array([0.5, 0.5])        # Initial guess
#     r = 1.0                     # Circle radius for projection
#     min = np.array([0, 0])      # Rectangle minimum
#     max = np.array([1, 1])      # Rectangle maximum

#     # Example usage for 3D case:
#     A_3d = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])  # 3D positive definite matrix
#     b_3d = np.array([-2, -5, -3])  # Linear term for 3D
#     x0_3d = np.array([0.5, 0.5, 0.5])  # Initial guess for 3D
