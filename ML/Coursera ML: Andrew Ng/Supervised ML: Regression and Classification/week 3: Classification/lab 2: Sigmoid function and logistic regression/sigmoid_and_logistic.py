import numpy as np
import matplotlib.pyplot as plt
from plt_one_addpt_onclick import plt_one_addpt_onclick
from lab_utils_common import draw_vthresh
plt.style.use('./deeplearning.mplstyle')


def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g

z_tmp = np.arange(-10,11)

y = sigmoid(z_tmp)

np.set_printoptions(precision=3)
print("Input (z), Output (sigmoid(z))")
print(np.c_[z_tmp, y])

# plot sigmoid function
fig, ax = plt.subplots(1, 1, figsize=(5,3))
ax.plot(z_tmp, y, c="b")

ax.set_title("Sigmoid function")
ax.set_ylabel('sigmoid(z)')
ax.set_xlabel('z')
draw_vthresh(ax, 0)


x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])

w_in = np.zeros((1))
b_in = 0

plt.close('all')
addpt = plt_one_addpt_onclick(x_train, y_train, w_in, b_in, logistic=True)
plt.show()
