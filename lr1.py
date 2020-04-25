import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def analytical_method(C):
    u = np.ones(len(C))
    C_rev = np.linalg.inv(C)
    v = 1 / (u @ C_rev @ u)
    x = (u @ C_rev) * v
    y = (C_rev @ u) * v
    print("Аналитический метод",
         f"Стратегии игрока А: {x}",
         f"Стратегии игрока В: {y}",
         f"Цена игры: {v}", sep='\n')

def iterative_method(C, e=0.1, first_x=1, first_y=1):
    a_counts = np.zeros(3)
    b_counts = np.zeros(3)
    # Первый ход
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

    print(df)
    df.to_csv('iter.csv')
    print("Итеративный метод",
         f"Стратегии игрока А: {a_counts/k}",
         f"Стратегии игрока В: {b_counts/k}",
         f"Цена игры: {(v_max + v_min)/2}", sep='\n')

    # график погрешности
    df['e'].plot()
    # графики верхней и нижней цен игры
    df[['v^', 'v_']].plot()
    plt.show()



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

if __name__ == '__main__':
    e = 0.1
    C = np.array([
        [9, 10, 13],
        [1, 18, 11],
        [17, 4, 0]
    ])
    analytical_method(C)
    iterative_method(C, e)