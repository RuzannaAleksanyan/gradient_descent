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

def gradient_descent(A, b, x0, n, epsilon, epsilon0, method='standard'):
    x = x0
    k = 0
    max_iterations = 10000
    
    while k < max_iterations:
        grad = gradient(x, A, b)
        h_k = -grad  

        if np.linalg.norm(grad) < epsilon0:
            break

        numerator = np.dot(grad, h_k)  # f'(x^k)h^k
        denominator = np.dot(np.dot(A, h_k), h_k)  # (Ah^k, h^k)

        if method == 'standart':
            alpha = 1 / (1 + k)

        if method == 'fastest_descent_mode': 
            if denominator == 0: 
                alpha = 1e-10 
            else:
                alpha = numerator / denominator

        if method == 'sharing':
            # ????
            alpha = 1

        x = x + alpha * h_k 

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

    fastest_descent_mode_x, fastest_descent_mode_steps = gradient_descent(A, b, x0, n, epsilon, epsilon0, method='fastest_descent_mode')

    standart_x, standart_steps = gradient_descent(A, b, x0, n, epsilon, epsilon0, method='standart')

    sharing_x, sharing_steps = gradient_descent(A, b, x0, n, epsilon, epsilon0, method='sharing')

    # True minimum point
    x_star = -np.linalg.inv(A).dot(b)

    print(f'Gradient descent by standard method: x = {fastest_descent_mode_x}, steps = {fastest_descent_mode_steps}')
    print(f'Gradient descent by fastest descent method: x = {standart_x}, steps = {standart_steps}')
    print(f'Gradient descent by sharing method: x = {sharing_x}, steps = {sharing_steps}')

    # print(f'True minimum point: x* = {x_star}')
