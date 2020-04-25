from functools import partial
import numpy as np
import pandas as pd
import sympy


def H(a, b, c, d, e, x, y):
    return a*x**2 + b*y**2 + c*x*y + d*x + e*y

def calc_x(C, b_counts, k):
    a_win = C @ b_counts
    return {
        'A': a_win.argmax() + 1,
        'a_win': a_win,
        'v^': a_win.max() / k
    }

def calc_y(C, a_counts, k):
    b_loss = a_counts @ C
    return {
        'B': b_loss.argmin() + 1,
        'b_loss': b_loss,
        'v_': b_loss.min() / k
    }

def calc_row(C, a_counts, b_counts, k):
    row = calc_x(C, b_counts, k)
    row.update(calc_y(C, a_counts, k))
    return row

def iterative_method(C, e=0.1, first_x=None, first_y=None):
    a_counts = np.zeros(C.shape[0])
    b_counts = np.zeros(C.shape[1])
    # Первый ход
    first_x = first_x or np.random.randint(0, C.shape[0])
    first_y = first_y or np.random.randint(0, C.shape[1])

    a_counts[first_x-1] += 1
    b_counts[first_y-1] += 1

    v_min = 0
    v_max = np.inf

    cur_e = np.inf

    df = pd.DataFrame(columns=('k', 'A', 'B', 'a_win', 'b_loss', 'v^', 'v_', 'e'))
    df = df.set_index('k')

    k = 1

    while e < cur_e:
        row = calc_row(C, a_counts, b_counts, k)
        if v_min < row['v_']:
            v_min = row['v_']
        if v_max > row['v^']:
            v_max = row['v^']
        cur_e = (v_max - v_min)
        row['e'] = cur_e
        df.loc[k] = row
        a_counts[row['A']-1] += 1
        b_counts[row['B']-1] += 1
        k += 1
    return a_counts/k, b_counts/k

def check_saddle(C):
    # find max min
    minmax = C.min(1).max()
    maxmin = C.max(0).min()
    if minmax != maxmin:
        return None
    minmax_is = np.argwhere(C.min(1) == C.min(1).max()).reshape(-1)
    maxmin_js = np.argwhere(C.max(0) == C.max(0).min()).reshape(-1)
    for i in minmax_is:
        for j in maxmin_js:
            if C[i][j] == minmax:
                return (i,j)


if __name__ == "__main__":
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    a = -6
    b = 32/5
    c = 16
    d = -16/5
    e = -64/5

    H = partial(H, a, b, c, d, e)

    x, y = sympy.symbols('x, y')
    res = sympy.linsolve([2*a*x+c*y+d, 2*b*y+c*x+e], (x, y))
    for x, y in res:
        if x >=0 and y >= 0:
            x_ = x
            y_ = y
    H_ = H(x_, y_)
    print('Аналитическое решение: ', f'x={x_}, y={y_}, H(x,y)={H_}', sep='\n')

    N = 2
    while(True):
        C = np.fromfunction(lambda i, j: H(i/N, j/N), (N+1, N+1))
        print(f'N={N}', C, sep='\n')
        saddle = check_saddle(C)
        if saddle:
            print('Есть седловая точка:')
            x = saddle[0]/N
            y = saddle[1]/N
        else:
            print('Седловой точки нет, решение методом Брауна-Робинсон:')
            a, b = iterative_method(C)
            distr = np.fromfunction(lambda i: i/N, (N+1,))
            x = a @ distr
            y = b @ distr
        print(f'x={x}, y={y}, H={H(x,y)}')
        if abs(H(x, y) - H_) < 0.001:
            break
        N += 1

    print('Найдено решение:', 
         f'x={x}, y={y}, H={H(x,y)}', 
         f'Величина погрешности: {abs(H(x, y) - H_)}',
         sep='\n'
    )
