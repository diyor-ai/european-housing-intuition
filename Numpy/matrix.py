import numpy as np
import time

# 1. Matritsalarni yaratish (2x2 dan boshla, keyin 500x500 ga o'zgartir)
size = 3
A = np.random.rand(size, size, size, size)
B = np.random.rand(size, size, size, size)

print(f"A matrisa {A}")
print(f"B matrisa {B}")
# # --- 1-usul: For loop (Eski va sekin usul) ---
# start_for = time.time()
# C_for = np.zeros((size, size))
# # Diqqat: Bu qism 500x500 da bir necha soniya olishi mumkin!
# for i in range(size):
#     for j in range(size):
#         for k in range(size):
#             C_for[i][j] += A[i][k] * B[k][j]
# print(f"For-loop vaqti: {time.time() - start_for:.4f} sek")

# --- 2-usul: NumPy (Zamonaviy va tezkor) ---
start_np = time.time()
C_np = A @ B  # @ belgisi np.dot(A, B) bilan bir xil
print(f"NumPy vaqti: {time.time() - start_np:.4f} sek")

# # Tekshirish: Ikkala natija bir xilmi?
# print(f"Natijalar bir xilmi? {np.allclose(C_for, C_np)}")