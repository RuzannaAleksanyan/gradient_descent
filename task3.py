# էջ 114
# Կազմել ծրագիր, որը իրականացնում է f(x)=1/2(Ax, a) = (b, x) 
# ֆունկցիայի մինիմիզացիան M = {x պատկանում է R^n | Cx <= d, x >= 0} 
# բազմության վրա պայմանական գրադիենտի եթոդով։ Այնտեղ A-ն (nxn) 
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

if __name__ == "__main__":
    n = 5
    m = 3
    A = np.random.randn(n, n)
    A = A.T @ A  # Make A symmetric positive definite
    b = np.random.randn(n)
    C = np.random.randn(m, n)
    d = np.random.randn(m)

    epsilon_0 = 1e-6
    x_opt = conditional_gradient_method(A, b, C, d, epsilon_0)
    print("Optimal solution:", x_opt)
