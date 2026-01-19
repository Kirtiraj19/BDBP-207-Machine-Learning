# Implement y = 2x1 + 3 and plot x1, y [start=-100, stop=100, num=100]

x_values = []
y_values = []

start = -100
stop = 100
num = 100

step = (stop - start) / num
x = start

while x <= stop:
    y = 2 * x + 3         #this is formula y = 2x+3
    x_values.append(x)    #store x value
    y_values.append(y)    #store y value
    x += step            # increase x by step size

# Print first 5 values
for i in range(5):
    print("x =", x_values[i], "y =", y_values[i])


#Output
# x = -100 y = -197
# x = -98.0 y = -193.0
# x = -96.0 y = -189.0
# x = -94.0 y = -185.0
# x = -92.0 y = -181.0