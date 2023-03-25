import numpy as np
from scipy.optimize import minimize
import time


# 目标函数
def fun(args1):
    b1, b2, b3, d1, d2, d3, pt, sc, x1, x2, x3 = args1
    r = lambda x: (b1 / x[0] + b2 / x[1] + b3 / x[2] + x1 * d1 / x[3] + x2 * d2 / x[4] + x3 * d3 / x[5] + (1 - x1) * (
                d1 / sc + pt) + (1 - x2) * (d2 / sc + pt) + (1 - x3) * (d3 / sc + pt))
    return r


def con(args2):
    w,s = args2
    cons = ({'type': 'eq', 'fun': lambda x: -x[0]-x[1]-x[2]+w},
            {'type': 'eq', 'fun': lambda x: -x[3]-x[4]-x[5]+s},
            {'type': 'ineq', 'fun': lambda x: x[0]-0.000001},
            {'type': 'ineq', 'fun': lambda x: x[1]-0.000001},
            {'type': 'ineq', 'fun': lambda x: x[2]-0.000001},
            {'type': 'ineq', 'fun': lambda x: x[3]-0.000001},
            {'type': 'ineq', 'fun': lambda x: x[4]-0.000001},
            {'type': 'ineq', 'fun': lambda x: x[5]-0.000001})
    return cons


def main():
    args1 = (3, 2, 1.5, 10, 13, 4.885820, 1.103162, 5, 1, 1, 0)
    args2 = (7, 10)
    cons = con(args2)
    x0 = np.array((2.3, 2.3, 2.3,3.3,3.3,3.3))  # 初值
    # res = minimize(fun(args1), x0, method='SLSQP', constraints=cons)
    start = time.time()
    res = minimize(fun(args1), x0, constraints=cons)
    end = time.time()
    print("time="+str(end-start))
    # print('minf(x):', res.fun)
    # print(res.success)
    # print('x：', [np.around(i) for i in res.x])
    # # 另一种表述
    print("optimization problem(res):{}".format(res.x))
    print("Xopt={}".format(res.x))
    print("minf(x)={:.4f}".format(res.fun))


if __name__ == "__main__":
    main()
