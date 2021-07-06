import matplotlib.pyplot as plt
import numpy as np
from numpy.random import random


def sgd_momentum(theta, vel, loss_constant=False, alpha=1, beta=0.9):
    derivative_loss = 10
    if not loss_constant:
        derivative_loss = random()

    vel = beta * vel + alpha * derivative_loss
    return theta + vel, vel


theta = 0
v = 0
velocity_history = []

for _ in range(10000):
    theta, v = sgd_momentum(theta, v, beta=1)
    velocity_history += [v]

plt.plot(velocity_history, label='velocity')
plt.legend()
plt.xlabel('time')
plt.show()
