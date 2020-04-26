import numpy as np
import pandas as pd
from numpy.lib.recfunctions import unstructured_to_structured as beautify

def crossroad(d1=0, d2=0 , e=0.001):
    return np.array([
        [(1-d1, 1-d2), (1-e-d1, 2)],
        [(2, 1-e-d2), (0, 0)]
    ])

def bos():
    return np.array([
        [(4,1), (0,0)],
        [(0,0), (1,4)]
    ])

def prisoners_d():
    return np.array([
        [(-5,-5), (0,-10)],
        [(-10,0), (-1,-1)]
    ])

def rand_matrix(shape=(10,10), lim=50):
    return np.stack([np.random.randint(-lim, lim, size=shape) for _ in range(2)], axis=-1)

def find_nash_equilibrium(C):
    A = C[:,:,0]
    B = C[:,:,1].T
    
    A_nash = A.max(0) == A
    B_nash = B.max(0) == B

    return A_nash & B_nash.T

def check_cell_pareto(C, i, j):
    val = C[i, j]
    weak = np.logical_and.reduce(C>=val, -1)
    strong = np.logical_or.reduce(C>val, -1)
    is_pareto_dominanated = np.logical_and(weak, strong).any()
    return not is_pareto_dominanated

def find_pareto_optima(C):
    res = np.ndarray((C.shape[0], C.shape[1]), bool)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            res[i,j] = check_cell_pareto(C, i, j)
    return res

def string_res(C, res):
    if np.argwhere(res).size == 0:
        return 'Пустое множество'
    return '\n'.join([f'i={x} j={y} u={tuple(C[x,y])}' for x, y in np.argwhere(res)])

if __name__ == "__main__":
    # crossroad_consts
    e =  np.round(np.random.random() * 0.3, 3)
    d1 = np.round(np.random.random() * 0.7, 3)
    d2 = np.round(np.random.random() * 0.7, 3)
    first_task = [
        ('Случайная игра (10х10):', rand_matrix()),
        ('Семейный спор', bos()),
        (f'Перекрёсток, e={e}, d1={d1}, d2={d2}', crossroad(d1, d2, e)),
        ('Дилемма заключённого', prisoners_d())
    ]
    df = pd.DataFrame(beautify(first_task[0][1]).astype('O'))
    df.to_csv('random_matrix.csv')
    for title, C in first_task:
        ne = find_nash_equilibrium(C)
        po = find_pareto_optima(C)
        print(title, 
        beautify(C), 
        'Равновесие Нэша:', 
        string_res(C, ne),
        'Оптимум по Парето:',
        string_res(C, po),
        'Пересечение:',
        string_res(C, np.logical_and(ne, po)),
        '',
        sep='\n')   
    
    C = np.array([
        [(3,0), (5,4)],
        [(11,6), (6,7)]
    ])
    print('Дана биматричная игра (2х2):', beautify(C), sep='\n')
    ne = find_nash_equilibrium(C)
    if len(np.argwhere(ne)) == 1:
        print('Игра и её смешанное расширение имеют единственную ситуацию равновесия по Нэшу:', string_res(C, ne), sep='\n')
    elif len(np.argwhere(ne)) == 0:
        print('Игра не имеет ситуацию равновесия по Нэшу в чистых стратегиях.')
    else:
        print('Игра имеет две равновесные по Нэшу ситуации:', string_res(C, ne), 'Вполне смешанная ситуация равновесия:', sep='\n')