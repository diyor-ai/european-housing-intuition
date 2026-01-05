import numpy as np

vec1 = np.array([4,5,6])
vec2 = np.array([7,8,9])

sum_vec = vec1 + vec2
print("Sum of two vecors is " ,sum_vec)

prod_vec = vec1 * vec2
print("Product of two vectors is " ,prod_vec)

dot_prod = np.dot(vec1, vec2)
print("Dot product is " ,dot_prod)

norm_vec = np.linalg.norm(vec1)
print("Norm of the vector is " ,norm_vec)