from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from math import pi, acos

class Ellipsoid:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def get_point(self, theta, phi):
        # theta - the angle of the projection in the xy-plane
        # phi - the angle from the polar axis, ie the polar angle
        # Transformation formulae for a spherical coordinate system.
        x = self.a*np.sin(theta)*np.cos(phi)
        y = self.b*np.sin(theta)*np.sin(phi)
        z = self.c*np.cos(theta)
        return np.array([x, y, z])

    def rand_point(self):
        theta = acos(2 * np.random.random() - 1)
        phi = np.random.random() * 2 * pi
        return self.get_point(theta, phi)

    def show_figure(self, points=None, target_point=None):
        theta = np.linspace(0, np.pi, 20)
        phi = np.linspace(0,2*np.pi, 40)
        T, P = np.meshgrid(theta, phi)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        X, Y, Z = self.get_point(T, P)
        ax.plot_wireframe(X, Y, Z, color='k')
        points = points or []
        for point in points:
            ax.scatter(*point, c='red', zorder=10)
        if target_point is not None:
            ax.scatter(*target_point, c='green', zorder=10)
        plt.show()

    def check_point(self, points, new_point, r):
        for point in points:
            dist = np.linalg.norm(point - new_point)
            if dist <= r:
                return False
        return True

    

def game(e, s, r, n_rounds, k=2, show_flag=False):
    points = []

    while len(points) != s:
        new_point = e.rand_point()
        if e.check_point(points, new_point, k * r):
            points.append(new_point)

    wins = 0
    for i in range(n_rounds):
        sp_point = e.rand_point()
        if not e.check_point(points, sp_point, r):
            wins += 1

        if not i and show_flag:
            e.show_figure(points, sp_point)


    return wins / n_rounds

if __name__ == '__main__':
    # Параметры
    a = 1   
    b = 2
    c = 3

    s = 500
    r = 0.1

    n_games = 10
    n_rounds = 100

    e = Ellipsoid(a, b, c)
    show = True #set to True to show plot
    # show = False

    print('Параметры игры:', f'a={a}, b={b}, c={c}', f's={s}, r={r}', sep='\n')
    

    g_costs = []
    for i in range(n_games):
        show = show and True if not i else False
        g_costs.append(game(e, s, r, n_rounds, 2, show))
        if i%(n_games/10) == 0:
            print(f'Проведена игра №{i+1} с ценой игры {g_costs[-1]}')

    print(f'Проведено {n_games} игр с {n_rounds} раундов в каждой')
    print(f'Средняя цена игры по всем играм: {sum(g_costs)/len(g_costs)}')
    

