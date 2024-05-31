import numpy as np

def f(data):
    x = data[0]
    y = data[1]
    part1 = 1 + pow((1.5 - x + x*y), 2)
    part2 = pow((2.25 - x + x*y**2), 2)
    part3 = pow((2.625 - x + x*y**3), 2)

    return np.log(part1 + part2 + part3) / 10
