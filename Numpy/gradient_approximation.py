def f(x):
    return x ** 2


def get_gradient(x):
    h = 0.0001
    return (f(x + h) - f(x)) / h


point_1 = 3.0
point_2 = -2
point_3 = 0
gradient_1 = get_gradient(point_1)
gradient_2 = get_gradient(point_2)
gradient_3 = get_gradient(point_3)

print(f"x = {point_1} nuqtadagi gradient: {gradient_1}")
print(f"x = {point_2} nuqtadagi gradient: {gradient_2}")
print(f"x = {point_3} nuqtadagi gradient: {gradient_3}")

# Insight: Gradient funksiya o'zgarish tezligini ko'rsatadi — AI'da loss kamayish yo'nalishi. x=0 da 0, chunki minimum.
# Bias check: Agar data biased bo'lsa (masalan, faqat positive x), gradient noto'g'ri yo'nalish beradi — diverse data kerak.
