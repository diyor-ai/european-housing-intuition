import time

import numpy as np

# 1. Matritsalarni yaratish (2-D, size x size)
size = 200
rng = np.random.default_rng(42)
A = rng.random((size, size))
B = rng.random((size, size))

if size <= 5:
    print(f"A matrisa:\n{A}")
    print(f"B matrisa:\n{B}")

# --- 1-usul: For loop (eski va sekin usul) ---
start_for = time.perf_counter()
C_for = np.zeros((size, size))
for i in range(size):
    for j in range(size):
        for k in range(size):
            C_for[i][j] += A[i][k] * B[k][j]
time_for = time.perf_counter() - start_for
print(f"For-loop vaqti: {time_for:.4f} sek")

# --- 2-usul: NumPy (zamonaviy va tezkor) ---
start_np = time.perf_counter()
C_np = A @ B  # @ belgisi matritsa ko'paytmasi (np.matmul)
time_np = time.perf_counter() - start_np
print(f"NumPy vaqti: {time_np:.6f} sek")

# Tekshirish: ikkala natija bir xilmi?
print(f"Natijalar bir xilmi? {np.allclose(C_for, C_np)}")
print(f"NumPy taxminan {time_for / time_np:,.0f}x tezroq")
