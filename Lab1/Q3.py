#Implement y = 2x12 + 3x1 + 4 and plot x1, y in the range [start=--10, stop=10, num=100]

x_values = []
y_values = []

x = -10
step = 0.2

while x <= 10:
    y = 2 * (x ** 2) + 3 * x + 4
    x_values.append(x)
    y_values.append(y)
    x += step

for i in range(5):
    print("x =", x_values[i], "y =", y_values[i])
