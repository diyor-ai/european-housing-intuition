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

House_A = np.array([120,3,400000])

new_House = House_A * 2

print("House A is " ,new_House)

arr = np.array([1,2,3,4,5])
print(arr)

print(type(arr))

dior = np.array([2, 4, 6]) * np.array([2, 4, 7])
print(dior)
