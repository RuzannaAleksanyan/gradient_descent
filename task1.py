# էջ 81
# Գրել ծրագիր, որը մինիմիզացնում է f(x)=1/2*(Ax,x)+(b,x)  
# ֆունկցիան R^n-ի վրա։ Այստեղ A-ն (n x n) չափանի սիմետրիկ, 
# դրական որոշյալ մատրից է, իսկ b֊ն n չափանի վեկտոր է։ 
# Մուտքի տվյալներն են A, b, x^0, n, epsilon, epsilon0 պարամետրերը։ 
# Այստեղ  epsilon0֊ն ճշտությունն է։ Կառուցել {x^k} հաջորդականությունը 
# գրադիենտային իջեցման 3 մեթոդներով և համեմատել դրանք քայլերի քանակի տեսակետից։ 
# Եթե ||f'(x^k)|| < epsilon0 , ապա պրոցեսն ավարտել և համարել x^k-ն մինիմումի կետ։ 
# Համեմատել ստացված արդյունքները x*=-A^(-1)b վեկտորի հետ, որը f-ի մինիմումի կետն է։

import numpy as np

def gradient(x, A, b):
    return np.dot(A, x) + b

# A = 1/2(Ax, x) + (b, x)
def f(x, A, b):
    return 0.5 * np.dot(x.T, np.dot(A, x)) + np.dot(b.T, x)

def gradient_descent(A, b, x0, epsilon, epsilon0, method='standard'):
    x = x0
    k = 0
    max_iterations = 10000
    
    while k < max_iterations: 
        grad = gradient(x, A, b)
        h_k = -grad  # descent direction

        # Stop condition
        if np.linalg.norm(grad) < epsilon0:
            break

        if method == 'standard':
            alpha = 1 / (1 + k)

        elif method == 'fastest_descent':
            numerator = np.dot(grad, h_k)
            denominator = np.dot(np.dot(A, h_k), h_k)
            alpha = numerator / denominator if denominator != 0 else 1e-10

        elif method == 'sharing':
            alpha = 1
            while f(x + alpha * h_k, A, b) > f(x, A, b) - epsilon * alpha * np.linalg.norm(grad)**2:
                alpha *= 0.5

        x = x + alpha * h_k  # Update the point

        k += 1
    
    return x, k


def is_symmetric(matrix):
    return np.array_equal(matrix, matrix.T)

def is_positive_semi_definite(matrix):
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

def input_matrix(n):
    print(f"Enter the values for a {n}x{n} matrix (row by row):")
    A = []
    for i in range(n):
        while True:
            try:
                row = list(map(float, input(f"Row {i + 1}: ").split()))
                if len(row) != n:
                    print(f"Error: Row {i + 1} must contain exactly {n} elements.")
                    continue
                A.append(row)
                break
            except ValueError:
                print("Error: Please enter valid numerical values.")
    
    A = np.array(A)
    print("Entered Matrix A:")
    print(A)

    if not is_symmetric(A):
        print("The matrix is not symmetric. Please ensure A[i,j] == A[j,i].")
        exit()
    
    if not is_positive_semi_definite(A):
        print("The matrix is not positive semi-definite. Please check the values entered.")
        exit()
    
    return A

def input_vector(n):
    print(f"Enter the values for a vector of length {n} (negative values allowed):")
    while True:
        try:
            # Split input into a list of floats
            vector = list(map(float, input("Vector: ").split()))
            # Check if the length matches the expected size
            if len(vector) != n:
                print(f"Error: The vector must contain exactly {n} elements.")
                continue
            # Return the vector as a numpy array
            return np.array(vector)
        except ValueError:
            # Handle non-numeric input
            print("Error: Please enter valid numerical values.")


# Main execution
if __name__ == "__main__":
    n = int(input("Enter the size of the matrix A (n): "))
    A = input_matrix(n)  # User inputs the matrix
    b = input_vector(n)   # User inputs the vector
    x0 = np.zeros(n)      # A one-dimensional array of length n
    epsilon = 1e-6        # Convergence tolerance
    epsilon0 = 1e-6       # Gradient tolerance

    standard_x, standart_steps = gradient_descent(A, b, x0, epsilon, epsilon0, method='standard')

    fastest_descent_x, fastest_descent_steps = gradient_descent(A, b, x0, epsilon, epsilon0, method='fastest_descent')

    sharing_x, sharing_steps = gradient_descent(A, b, x0, epsilon, epsilon0, method='sharing')

    # True minimum point
    x_star = -np.linalg.inv(A).dot(b)

    print(f'Gradient descent by standard method: x = {fastest_descent_x}, steps = {fastest_descent_steps}')
    print(f'Gradient descent by fastest descent method: x = {standard_x}, steps = {standart_steps}')
    print(f'Gradient descent by sharing method: x = {sharing_x}, steps = {sharing_steps}')

    print(f'True minimum point: x* = {x_star}')

    print(".........................................................")

    # Optional: Calculate distances to true minimum
    distance_standard = np.linalg.norm(standard_x - x_star)
    distance_fastest_descent = np.linalg.norm(fastest_descent_x - x_star)
    distance_sharing = np.linalg.norm(sharing_x - x_star)

    # Print distances to the true minimum
    print(f'Distance between standard descent solution and x*: {distance_standard}')
    print(f'Distance between fastest descent solution and x*: {distance_fastest_descent}')
    print(f'Distance between sharing descent solution and x*: {distance_sharing}')


    # # True minimum point
    # x_star = -np.linalg.inv(A).dot(b)

    # distance_standard = np.linalg.norm(standart_x - x_star)
    # distance_fastest_descent = np.linalg.norm(fastest_descent_x - x_star)
    # distance_sharing = np.linalg.norm(sharing_x - x_star)

    # # Print the comparison results
    # print(f'Distance between standard descent solution and x*: {distance_standard}')
    # print(f'Distance between fastest descent solution and x*: {distance_fastest_descent}')
    # print(f'Distance between sharing descent solution and x*: {distance_sharing}')
    
    # # Find the closest solution to x_star
    # if distance_fastest_descent < distance_standard and distance_fastest_descent < distance_sharing:
    #     closest_solution = "Fastest Descent"
    #     closest_value = fastest_descent_x
    #     min_distance = distance_fastest_descent
    # elif distance_standard < distance_fastest_descent and distance_standard < distance_sharing:
    #     closest_solution = "Standard Descent"
    #     closest_value = standart_x
    #     min_distance = distance_standard
    # else:
    #     closest_solution = "Sharing Descent"
    #     closest_value = sharing_x
    #     min_distance = distance_sharing

    # # Print the results
    # print(f"The closest solution to x* is from the {closest_solution} method.")
    # print(f"Solution: {closest_value}")
    # print(f"Distance to x*: {min_distance}")
