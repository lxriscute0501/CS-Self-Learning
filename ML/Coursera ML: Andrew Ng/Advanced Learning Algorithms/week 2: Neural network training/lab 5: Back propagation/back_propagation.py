from lab_utils_backprop import *

plt.close("all")
plt_network(config_nw0, "./images/C2_W2_BP_network0.PNG")

w = 3
a = 2 + 3 * w
J = a ** 2
print(f"a = {a}, J = {J}")

a_epsilon = a + 0.001
J_epsilon = a_epsilon ** 2
k = (J_epsilon - J) / 0.001
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_da ~= k = {k} ")

sw, sJ, sa = symbols('w,J,a')
sJ = sa ** 2
print(sJ)

sJ.subs([(sa,a)])

dJ_da = diff(sJ, sa)
print(dJ_da)


w_epsilon = w + 0.001
a_epsilon = 2 + 3 * w_epsilon
k = (a_epsilon - a) / 0.001
print(f"a = {a}, a_epsilon = {a_epsilon}, da_dw ~= k = {k} ")

sa = 2 + 3 * sw
print(sa)

da_dw = diff(sa, sw)
print(da_dw)

dJ_dw = da_dw * dJ_da
print(dJ_dw)

w_epsilon = w + 0.001
a_epsilon = 2 + 3 * w_epsilon
J_epsilon = a_epsilon ** 2
k = (J_epsilon - J) / 0.001
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k} ")


plt.close("all")
plt_network(config_nw1, "./images/C2_W2_BP_network1.PNG")

# Inputs and parameters
x = 2
w = -2
b = 8
y = 1
# calculate per step values
c = w * x
a = c + b
d = a - y
J = d ** 2 / 2
print(f"J={J}, d={d}, a={a}, c={c}")

d_epsilon = d + 0.001
J_epsilon = d_epsilon ** 2 / 2
k = (J_epsilon - J) / 0.001
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dd ~= k = {k} ")

sx, sw, sb, sy, sJ = symbols('x,w,b,y,J')
sa, sc, sd = symbols('a,c,d')
sJ = sd ** 2 / 2
print(sJ)

sJ.subs([(sd,d)])

dJ_dd = diff(sJ, sd)
print(dJ_dd)

a_epsilon = a + 0.001
d_epsilon = a_epsilon - y
k = (d_epsilon - d) / 0.001
print(f"d = {d}, d_epsilon = {d_epsilon}, dd_da ~= k = {k} ")

sd = sa - sy
print(sd)

dd_da = diff(sd, sa)
print(dd_da)

dJ_da = dd_da * dJ_dd
print(dJ_da)

a_epsilon = a + 0.001
d_epsilon = a_epsilon - y
J_epsilon = d_epsilon ** 2 / 2
k = (J_epsilon - J) / 0.001
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_da ~= k = {k} ")

sa = sc + sb
print(sa)

da_dc = diff(sa,sc)
da_db = diff(sa,sb)
print(da_dc, da_db)

dJ_dc = da_dc * dJ_da
dJ_db = da_db * dJ_da
print(f"dJ_dc = {dJ_dc},  dJ_db = {dJ_db}")

sc = sw * sx
print(sc)

dc_dw = diff(sc, sw)
print(dc_dw)

dJ_dw = dc_dw * dJ_dc
print(dJ_dw)

print(f"dJ_dw = {dJ_dw.subs([(sd,d),(sx,x)])}")

J_epsilon = ((w + 0.001) * x + b - y) ** 2 / 2
k = (J_epsilon - J) / 0.001
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k} ")