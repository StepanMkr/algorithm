import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Callable, List, Dict
import time


# Функция Розенброка
def rosenbrock(x: np.ndarray) -> float:
    x1, x2 = x
    return 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2

# Градиент функции Розенброка
def rosenbrock_gradient(x: np.ndarray) -> np.ndarray:
    x1, x2 = x
    df_dx1 = -400 * x1 * (x2 - x1 ** 2) - 2 * (1 - x1)
    df_dx2 = 200 * (x2 - x1 ** 2)
    return np.array([df_dx1, df_dx2])


# Линейный поиск с дроблением шага
def line_search(f: Callable, x: np.ndarray, d: np.ndarray,
                grad: np.ndarray, f_count: List[int]) -> float:

    alpha = 1.0
    rho = 0.5
    c = 1e-4
    f_current = f(x)
    f_count[0] += 1
    grad_d = grad @ d

    max_trials = 30
    trial = 0

    while trial < max_trials:
        f_new = f(x + alpha * d)
        f_count[0] += 1

        if f_new <= f_current + c * alpha * grad_d:
            return alpha

        alpha *= rho
        trial += 1

    return 1e-8


# Метод DFP
def dfp_optimize(f: Callable, grad: Callable, x0: np.ndarray,
                 max_iter: int = 1000, tol: float = 1e-6) -> Dict:
    """
    Минимизирует функцию f с помощью метода DFP.

    Args:
        f: целевая функция
        grad: градиент функции
        x0: начальная точка
        max_iter: максимальное число итераций
        tol: допустимая точность по норме градиента

    Returns:
        Словарь с результатами оптимизации
    """
    n = len(x0)
    x = x0.copy().astype(float)
    H = np.eye(n)  # начальное приближение гессиана

    # Счетчики
    f_count = [0]  # используем список для передачи по ссылке
    g_count = [0]
    start_time = time.time()

    # История для траектории
    history = [x.copy()]

    # Начальное значение функции
    f_current = f(x)
    f_count[0] += 1

    for i in range(max_iter):
        # Вычисляем градиент
        g = grad(x)
        g_count[0] += 1

        # Проверка сходимости
        if np.linalg.norm(g) < tol:
            break

        # Направление спуска
        p = -H @ g

        # Линейный поиск
        alpha = line_search(f, x, p, g, f_count)

        # Делаем шаг
        s = alpha * p
        x_new = x + s

        # Вычисляем значение функции в новой точке
        f_new = f(x_new)
        f_count[0] += 1

        # Вычисляем новый градиент
        g_new = grad(x_new)
        g_count[0] += 1
        y = g_new - g

        # Обновление H по формуле DFP
        sy = s @ y
        yHy = y @ H @ y

        if abs(sy) > 1e-16 and abs(yHy) > 1e-16:
            term1 = np.outer(s, s) / sy
            Hy = H @ y
            term2 = np.outer(Hy, Hy) / yHy
            H = H + term1 - term2

        x = x_new
        history.append(x.copy())

    end_time = time.time()

    # Вычисляем финальное значение градиента
    final_grad = grad(x)
    g_count[0] += 1

    return {
        'x_opt': x,
        'f_opt': f(x),
        'iterations': i + 1,
        'f_count': f_count[0],
        'g_count': g_count[0],
        'time': end_time - start_time,
        'converged': np.linalg.norm(final_grad) < tol,
        'final_gradient_norm': np.linalg.norm(final_grad),
        'history': np.array(history)
    }


# Функция для построения графика с траекториями
def plot_rosenbrock_with_trajectory(results: List[Dict], test_points: List[np.ndarray]):
    # Создаем сетку для контурного графика
    x1 = np.linspace(-2.5, 6.5, 200)
    x2 = np.linspace(-4, 5, 200)
    X1, X2 = np.meshgrid(x1, x2)

    # Вычисляем значения функции на сетке
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = rosenbrock(np.array([X1[i, j], X2[i, j]]))

    plt.figure(figsize=(12, 10))

    # Контурный график с логарифмическими уровнями
    levels = np.logspace(-1, 4, 30)
    contour = plt.contour(X1, X2, Z, levels=levels, cmap='viridis', alpha=0.7)
    plt.colorbar(contour, label='f(x) - лог. шкала')

    # Находим индекс сошедшейся точки
    converged_index = 0

    # Рисуем траекторию только для сошедшейся точки
    result = results[converged_index]
    point = test_points[converged_index]
    history = result['history']

    # Траектория метода (красная линия)
    plt.plot(history[:, 0], history[:, 1], 'red', linewidth=2,
             marker='o', markersize=4, markevery=max(1, len(history) // 5),
             label=f'Траектория DFP из точки ({point[0]:.3f}, {point[1]:.3f})')

    # Начальная точка (звезда)
    plt.plot(history[0, 0], history[0, 1], 'red', marker='*',
             markersize=20, markeredgecolor='black', markeredgewidth=1.5,
             label=f'Старт: ({point[0]:.3f}, {point[1]:.3f})')

    # Конечная точка (квадрат)
    plt.plot(history[-1, 0], history[-1, 1], 'red', marker='s',
             markersize=15, markeredgecolor='black', markeredgewidth=1.5,
             label=f'Финиш: ({result["x_opt"][0]:.3f}, {result["x_opt"][1]:.3f})')

    # Глобальный минимум
    plt.plot(1, 1, 'y*', markersize=25, label='Глобальный минимум (1, 1)',
             markeredgecolor='black', markeredgewidth=1.5)

    plt.xlabel('x1', fontsize=12)
    plt.ylabel('x2', fontsize=12)
    plt.title('Функция Розенброка и траектория метода DFP (сошедшаяся точка)', fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(-2.5, 6.5)
    plt.ylim(-4, 5)
    plt.tight_layout()
    plt.show()


def main():

    # Начальные вектора в формате (x1, x2)
    test_points = [
        np.array([1.200, 1.000]),
        np.array([-2.000, -2.000]),
        np.array([5.621, -3.635]),
        np.array([-0.221, 0.639])
    ]

    results = []

    # Запускаем оптимизацию для каждой начальной точки
    for i, x0 in enumerate(test_points):
        stats = dfp_optimize(rosenbrock, rosenbrock_gradient, x0,
                             max_iter=1000, tol=1e-6)

        results.append(stats)

    # Создаем сводную таблицу
    print("\n" + "=" * 90)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("=" * 90)

    table_data = []
    for i, (point, stats) in enumerate(zip(test_points, results)):
        table_data.append({
            '№': i + 1,
            'Начальная точка': f"({point[0]:.3f}, {point[1]:.3f})",
            'Найденный минимум': f"({stats['x_opt'][0]:.6f}, {stats['x_opt'][1]:.6f})",
            'f(x*)': f"{stats['f_opt']:.2e}",
            'Итерации': stats['iterations'],
            'Вызовы f(x)': stats['f_count'],
            'Вызовы ∇f': stats['g_count'],
            'Время (с)': f"{stats['time']:.4f}",
            '||∇f(x*)||': f"{stats['final_gradient_norm']:.2e}",
            'Сходимость': 'Да' if stats['converged'] else 'Нет'
        })

    df = pd.DataFrame(table_data)
    print(df.to_string(index=False))

    plot_rosenbrock_with_trajectory(results, test_points)


if __name__ == "__main__":
    main()