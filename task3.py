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
# x^(k+1) = x^k + alfak*h^k
# alfak֊ն ընտրում ենք կիսման եղանակով

import numpy as np
from scipy.optimize import linprog

# f(x) = 1/2 (Ax, x) - (b, x)
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
        raise ValueError(f"Linear programming (Simplex) did not converge: {result.message}")

def conditional_gradient_method(A, b, C, d, epsilon_0, max_iter=1000):
    n = len(b)
    
    # Initialize x^0 using the Simplex method
    x_k = simplex_method(np.zeros(n), C, d)
    
    k = 0
    while k < max_iter:
        grad = gradient(A, b, x_k)
        
        x_star = simplex_method(grad, C, d)
        
        h_k = x_star - x_k
        eta_k = np.dot(grad, h_k)
        
        # Stopping criterion
        if abs(eta_k) < epsilon_0:
            print(f"Converged after {k+1} iterations.")
            return k + 1, x_k
        
        # Perform line search to find alpha_k
        alpha_k = 1.0
        while f(x_k + alpha_k * h_k, A, b) > f(x_k, A, b) + epsilon_0 * alpha_k * eta_k:
            alpha_k *= 0.5
        
        # Update x^k
        x_k = x_k + alpha_k * h_k
        k += 1

    print("Max iterations reached.")
    return k, x_k 

def f(x, A, b):
    return 0.5 * np.dot(x.T, np.dot(A, x)) - np.dot(b.T, x)

def input_matrix_A(n):
    print(f"Enter the values for a {n}x{n} symmetric positive definite matrix A:")
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
    if not is_symmetric(A):
        print("The matrix is not symmetric. Please ensure A[i,j] == A[j,i].")
        exit()
    
    if not is_positive_definite(A):
        print("The matrix is not positive definite. Please check the values entered.")
        exit()
    
    return A

def is_symmetric(matrix):
    return np.allclose(matrix, matrix.T)

def is_positive_definite(matrix):
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

def input_matrix_C(m, n):
    print(f"Enter the values for a {m}x{n} constraint matrix C:")
    C = []
    for i in range(m):
        while True:
            try:
                row = list(map(float, input(f"Row {i + 1}: ").split()))
                if len(row) != n:
                    print(f"Error: Row {i + 1} must contain exactly {n} elements.")
                    continue
                C.append(row)
                break
            except ValueError:
                print("Error: Please enter valid numerical values.")
    return np.array(C)

def input_vector(length, name="vector"):
    print(f"Enter the values for the {name} of length {length}:")
    while True:
        try:
            vector = list(map(float, input(f"{name}: ").split()))
            if len(vector) != length:
                print(f"Error: The {name} must contain exactly {length} elements.")
                continue
            return np.array(vector)
        except ValueError:
            print("Error: Please enter valid numerical values.")

if __name__ == "__main__":
    # Input dimensions and matrices
    n = int(input("Enter the size of the matrix A (n): "))
    A = input_matrix_A(n)
    b = input_vector(n, name="b vector")

    m = int(input("Enter the number of constraints (m): "))
    C = input_matrix_C(m, n)
    d = input_vector(m, name="d vector")

    epsilon_0 = float(input("Enter the precision (epsilon_0): "))
    max_iter = int(input("Enter the maximum number of iterations: "))
    
    # Run the conditional gradient method
    step_count, x_opt = conditional_gradient_method(A, b, C, d, epsilon_0, max_iter)
    print("Optimal solution:", x_opt)
    print("Number of iterations:", step_count)
