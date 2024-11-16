# էջ 114
# Կազմել ծրագիր, որը իրականացնում է f(x)=1/2(Ax, a) = (b, x) 
# ֆունկցիայի մինիմիզացիան M = {x պատկանում է R^n | Cx <= d, x >= 0} 
# բազմության վրա պայմանական գրադիենտի մեթոդով։ Այնտեղ A-ն (nxn) 
# չափանի սիմետրիկ դրական որոշյալ մատրից է, իսկ C-ն (mxn) չափանի մատրից է։ 
# k-րդ քայլում օգտագործելով սիմպլեքս ալգորիթմը՝ ստանալ ettak = min(f'(x^k), x-x^k) 
# խնդրի որևէ լուծում։ Ալգորիթմի կանգառի համար ընդունել |ettak|<epsilon0 պայմանը, 
# որտեղ epsilon0>0 նախապես տրված ճշտություն է։ Եթե նշված պայմանը կատարվում է, 
# ապա x^k վեկտորը համարել խնդրի լուծում և ավարտել ալգորիթմը։ Սկզբնական x^0 
# պատկանում է M կետի ընտրությունը նույնպես կատարել սիմպլեքս ալգորիթմով։



import numpy as np
from scipy.optimize import linprog

def gradient(A, b, x):
    return np.dot(A, x) - b

def simplex_method(grad, C, d):
    c = grad
    bounds = [(0, None)] * len(c)
    A_ub = C
    b_ub = d
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if result.success:
        return result.x
    else:
        raise ValueError(f"Linear programming (Simplex) did not converge or is infeasible. {result.message}")

def conditional_gradient_method(A, b, C, d, epsilon_0, max_iter=1000):
    n = len(b)
    x_k = np.zeros(n)

    # Check if A is positive definite
    eigenvalues = np.linalg.eigvals(A)
    if np.any(eigenvalues <= 0):
        print("Warning: A is not positive definite.")
    
    # Check if initial x_k satisfies Cx <= d
    if np.any(np.dot(C, x_k) > d):
        print("Initial x_k does not satisfy the constraints. Adjusting x_k.")
        x_k = np.maximum(np.dot(np.linalg.pinv(C), d), 0)  # Adjust to feasible point
    
    print(f"Initial feasible x_k: {x_k}")
    
    for k in range(max_iter):
        grad = gradient(A, b, x_k)
        x_star = simplex_method(grad, C, d)
        eta_k = np.dot(grad, x_star - x_k) / np.dot(grad, grad)
        x_k = x_k + eta_k * (x_star - x_k)

        if abs(eta_k) < epsilon_0:
            print(f"Converged after {k+1} iterations.")
            return x_k
    
    print("Max iterations reached.")
    return x_k

def input_matrix_A(n):
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

def is_symmetric(matrix):
    return np.array_equal(matrix, matrix.T)

def is_positive_semi_definite(matrix):
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

def input_matrix_C(m, n):
    # print(f"Please enter the {m}x{n} matrix, with {m} rows and {n} columns.")
    # print(f"Enter the values row by row, separated by spaces.")

    matrix = []

    for i in range(m):
        while True:
            try:
                row_input = input(f"Enter row {i + 1} (space-separated values): ")
                row = list(map(float, row_input.strip().split()))

                if len(row) != n:
                    print(f"Error: You must enter exactly {n} values for row {i + 1}. Try again.")
                    continue

                matrix.append(row)
                break
            except ValueError:
                print("Error: Please enter valid numbers separated by spaces.")
    
    return np.array(matrix)

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


if __name__ == "__main__":
    n = int(input("Enter the size of the matrix A (n): "))
    A = input_matrix_A(n)  # User inputs the matrix

    m = int(input("Enter the dimension m of the matrix C: "))
    C = input_matrix_C(n, m)

    b = input_vector(n)

    # ?
    # d = np.random.randn(m)

    epsilon_0 = 1e-6
    # x_opt = conditional_gradient_method(A, b, C, d, epsilon_0)
    # print("Optimal solution:", x_opt)
