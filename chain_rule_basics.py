import numpy as np

def g(x):
    return x**2

def f(x):
    return 3*x

x = 2.0

dg_dx=2*x
df_dg = 3

dy_dx = df_dg*dg_dx

print("y =", f(g(x)))
print("dy/dx= ", dy_dx)

def composed(x):
    return f(g(x))

h = 1e-4
a = x + h
numerical_grad = (composed(x + h) - composed(x)) / h

print("Numerical gradient:", numerical_grad)
# Numerical gradient — need to check for analytic gradients