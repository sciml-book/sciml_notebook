import numpy as np
from numpy.polynomial import legendre
import os

def legendre_roots(q):
    """Find roots of the Legendre polynomial of degree q"""
    # Get coefficients of the Legendre polynomial
    # Note: numpy's legendre polynomials are defined on [-1,1]
    # We need to transform from [0,1] to [-1,1] using x -> 2x-1
    coef = np.zeros(q + 1)
    coef[-1] = 1  # Coefficient of highest degree term
    poly = legendre.Legendre(coef)
    
    # Find roots and transform back to [0,1] interval
    roots = legendre.legroots(coef)
    roots = (roots + 1) / 2
    
    return np.sort(roots)

def lagrange_basis(x, nodes, i):
    """Compute the i-th Lagrange basis polynomial at point x"""
    n = len(nodes)
    result = 1.0
    for j in range(n):
        if j != i:
            result *= (x - nodes[j]) / (nodes[i] - nodes[j])
    return result

def compute_butcher_tableau(q):
    """Compute the Butcher tableau for q-point Gauss method"""
    # Get the nodes (roots of Legendre polynomial)
    c = legendre_roots(q)
    
    # Initialize Butcher tableau
    A = np.zeros((q, q))
    b = np.zeros(q)
    
    # Compute the A matrix and b vector
    for i in range(q):
        # Compute b (last row of the tableau)
        b[i] = integrate_lagrange(c, i, 0, 1)
        
        # Compute A matrix
        for j in range(q):
            A[j,i] = integrate_lagrange(c, i, 0, c[j])
            
    return c, A, b

def integrate_lagrange(nodes, i, a, b, num_points=1000):
    """Numerically integrate the i-th Lagrange basis polynomial from a to b"""
    x = np.linspace(a, b, num_points)
    y = np.array([lagrange_basis(xi, nodes, i) for xi in x])
    return np.trapz(y, x)

def write_butcher_file(q, filename):
    """Write Butcher tableau to file in the specified format"""
    c, A, b = compute_butcher_tableau(q)
    
    with open(filename, 'w') as f:
        # Write the A matrix row by row
        for i in range(q):
            for j in range(q):
                f.write(f"{A[i,j]:.18f}\n")
        
        # Write the b vector
        for bi in b:
            f.write(f"{bi:.18f}\n")
        
        # Write the c vector
        for ci in c:
            f.write(f"{ci:.18f}\n")

def main():
    # Create directory for output files
    os.makedirs("butcher_irk_files", exist_ok=True)
    os.chdir("butcher_irk_files")
    
    # Generate files for q = 1 to 500
    for q in range(1, 501):
        filename = f"Butcher_IRK{q}.txt"
        try:
            write_butcher_file(q, filename)
            if q % 10 == 0:
                print(f"Generated file for q = {q}")
        except Exception as e:
            print(f"Error generating file for q = {q}: {str(e)}")
    
    print("All files generated successfully!")

if __name__ == "__main__":
    main()