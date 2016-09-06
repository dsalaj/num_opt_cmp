from sympy import *
from sympy.parsing import sympy_parser as spp

import numpy as np
import matplotlib.pyplot as plt

init_printing(use_unicode=True)


# Plot range
plot_from, plot_to, plot_step = -7.0, 7.0, 0.1
# Precision for iterative methods
target_precision = 0.3

m = Matrix(symbols('x1 x2'))


def dfdx(x, g):
    return [float(g[i].subs(m[0], x[0]).subs(m[1], x[1])) for i in range(len(g))]


def sd(alpha=0.0002):
    """
    Steepest Descent - 1st order optimization
    :return:
    """
    print "STEEPEST DESCENT: start"
    # gradient
    g = [diff(obj, i) for i in m]
    # Initialize xs
    xs = [[0.0, 0.0]]
    xs[0] = x_start
    # Get gradient at start location (df/dx or grad(f))
    iter_s = 0
    while np.linalg.norm(xs[-1] - x_result) > target_precision:
        # print "STEEPEST DESCENT: distance:", np.linalg.norm(xs[-1] - x_result)
        gs = dfdx(xs[iter_s], g)
        # Compute search direction and magnitude (dx)
        #  with dx = - grad but no line searching
        xs.append(xs[iter_s] - np.dot(alpha, gs))
        # print xs[-1]
        iter_s += 1
        if iter_s > 10000:
            break
    print "STEEPEST DESCENT: result distance:", np.linalg.norm(xs[-1] - x_result)
    xs = np.array(xs)
    plt.plot(xs[:, 0], xs[:, 1], 'g-o')


def nm():
    """
    Newton's method - 2nd order optimization
    :return:
    """
    print "NEWTON METHOD: start"
    # gradient
    g = [diff(obj, i) for i in m]
    # Hessian matrix
    H = Matrix([[diff(g[j], m[i]) for i in range(len(m))] for j in range(len(g))])
    H_inv = H.inv()

    xn = [[0, 0]]  # Newton method result global for comparison
    xn[0] = x_start

    iter_n = 0
    while np.linalg.norm(xn[-1] - x_result) > target_precision:
        # print "NEWTON METHOD: distance:", np.linalg.norm(xn[-1] - x_result)
        gn = Matrix(dfdx(xn[iter_n], g))
        delta_xn = -H_inv * gn
        delta_xn = delta_xn.subs(m[0], xn[iter_n][0]).subs(m[1], xn[iter_n][1])
        xn.append(Matrix(xn[iter_n]) + delta_xn)
        iter_n += 1
    print "NEWTON METHOD: result distance:", np.linalg.norm(xn[-1] - x_result)

    xn = np.array(xn)
    plt.plot(xn[:, 0], xn[:, 1], 'k-o')


def es():
    """
    (1+1) Evolutionary Strategy - 0th order method
    :return:
    """
    print "EVOLUTIONARY STRATEGY: start"
    xe = [[0.0, 0.0]]
    xe[0] = x_start
    iter_e = 0
    n_good_mutations = 0.0
    e_step = 2
    n = 10

    while true:
        for i in range(n):
            new_xe = np.random.normal(xe[-1], e_step, 2)
            iter_e += 1
            if obj.subs(m[0], new_xe[0]).subs(m[1], new_xe[1]) < obj.subs(m[0], xe[-1][0]).subs(m[1], xe[-1][1]):
                n_good_mutations += 1
                xe.append(new_xe)

        distance = np.linalg.norm(xe[-1] - x_result)
        # print "EVOLUTIONARY STRATEGY: distance:", distance
        if distance < target_precision:
            break  # stopping criterion
        if iter_e >= n:
            p_pos = n_good_mutations / iter_e
            n_good_mutations = 0.0
            if p_pos < 0.2:
                e_step *= 0.85
            else:
                e_step /= 0.85
            iter_e = 0
    xe = np.array(xe)
    plt.plot(xe[:, 0], xe[:, 1], 'b-o')


if __name__ == '__main__':
    ####################
    # Quadratic function
    ####################
    # Start location
    x_start = [-4.0, 6.0]

    # obj = spp.parse_expr('x1**2 - x2 * x1 - x1 + 4 * x2**2')
    # x_result = np.array([16/15, 2/15])
    obj = spp.parse_expr('x1**2 - 2 * x1 * x2 + 4 * x2**2')
    x_result = np.array([0, 0])

    # Design variables at mesh points
    i1 = np.arange(plot_from, plot_to, plot_step)
    i2 = np.arange(plot_from, plot_to, plot_step)
    x1_mesh, x2_mesh = np.meshgrid(i1, i2)
    f_str = obj.__str__().replace('x1', 'x1_mesh').replace('x2', 'x2_mesh')
    f_mesh = eval(f_str)

    # Create a contour plot
    plt.figure()

    plt.imshow(f_mesh, cmap='Paired', origin='lower',
               extent=[plot_from - 20, plot_to + 20, plot_from - 20, plot_to + 20])
    plt.colorbar()

    # Add some text to the plot
    plt.title('f(x) = ' + str(obj))
    plt.xlabel('x1')
    plt.ylabel('x2')
    nm()
    sd(alpha=0.05)
    es()
    plt.show()

    #####################
    # Rosenbrock function
    #####################
    # Start location
    x_start = [-4.0, -5.0]

    obj = spp.parse_expr('(1 - x1)**2 + 100 * (x2 - x1**2)**2')
    x_result = np.array([1, 1])

    # Design variables at mesh points
    i1 = np.arange(plot_from, plot_to, plot_step)
    i2 = np.arange(plot_from, plot_to, plot_step)
    x1_mesh, x2_mesh = np.meshgrid(i1, i2)
    f_str = obj.__str__().replace('x1', 'x1_mesh').replace('x2', 'x2_mesh')
    f_mesh = eval(f_str)

    # Create a contour plot
    plt.figure()

    plt.imshow(f_mesh, cmap='Paired', origin='lower',
               extent=[plot_from - 20, plot_to + 20, plot_from - 20, plot_to + 20])
    plt.colorbar()

    # Add some text to the plot
    plt.title('f(x) = ' + str(obj))
    plt.xlabel('x1')
    plt.ylabel('x2')
    nm()
    sd(alpha=0.0002)
    es()
    plt.show()

    # import timeit
    # print(timeit.timeit("nm()", setup="from __main__ import nm", number=10))
    # print(timeit.timeit("sd()", setup="from __main__ import sd", number=10))
    # print(timeit.timeit("es()", setup="from __main__ import es", number=10))
