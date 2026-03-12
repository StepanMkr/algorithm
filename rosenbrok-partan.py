import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Callable, List, Dict, Tuple
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


# Линейный поиск (золотое сечение)
def line_search_golden(f: Callable, x: np.ndarray, d: np.ndarray,
                       f_count: List[int], alpha_max: float = 5.0) -> float:

    phi = (1 + np.sqrt(5)) / 2

    a, b = 0.0, alpha_max
    eps = 1e-6

    x1 = b - (b - a) / phi
    x2 = a + (b - a) / phi

    f1 = f(x + x1 * d)
    f2 = f(x + x2 * d)
    f_count[0] += 2

    while abs(b - a) > eps:
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = b - (b - a) / phi
            f1 = f(x + x1 * d)
            f_count[0] += 1
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + (b - a) / phi
            f2 = f(x + x2 * d)
            f_count[0] += 1

    return (a + b) / 2


# Модифицированный Партан-метод
def partan_optimize(f: Callable, grad: Callable, x0: np.ndarray,
                    max_iter: int = 1000, tol: float = 1e-6) -> Dict:
    """
    Минимизирует функцию f модифицированным Партан-методом.

    Алгоритм:
    1. Из точки X₀ делается два шага наискорейшего спуска → X₁ и X₂
    2. Одномерная оптимизация в направлении X₂ - X₀ → X₃
    3. Из X₃ шаг наискорейшего спуска → X₄
    4. Одномерная оптимизация из X₃ в направлении X₄ - X₂ → X₅
    5. Процесс повторяется с X₅
    """
    n = len(x0)
    x = x0.copy().astype(float)

    # Счетчики
    f_count = [0]
    g_count = [0]
    start_time = time.time()

    # История для траектории
    history = [x.copy()]
    function_values = [f(x)]
    f_count[0] += 1

    # Начальное значение градиента
    g = grad(x)
    g_count[0] += 1

    # Проверка сходимости в начальной точке
    if np.linalg.norm(g) < tol:
        end_time = time.time()
        return {
            'x_opt': x,
            'f_opt': function_values[-1],
            'iterations': 0,
            'f_count': f_count[0],
            'g_count': g_count[0],
            'time': end_time - start_time,
            'converged': True,
            'final_gradient_norm': np.linalg.norm(g),
            'history': np.array(history),
            'function_values': function_values
        }

    # Шаг 1 первый шаг наискорейшего спуска (х0 - х1)
    d1 = -g  # антиградиент
    alpha1 = line_search_golden(f, x, d1, f_count)
    x1 = x + alpha1 * d1
    history.append(x1.copy())

    # Вычисляем градиент в х1
    g1 = grad(x1)
    g_count[0] += 1
    function_values.append(f(x1))
    f_count[0] += 1

    # Шаг 2 второй шаг наискорейшего спуска х1 - х2
    d2 = -g1
    alpha2 = line_search_golden(f, x1, d2, f_count)
    x2 = x1 + alpha2 * d2
    history.append(x2.copy())

    # Вычисляем градиент в х2
    g2 = grad(x2)
    g_count[0] += 1
    function_values.append(f(x2))
    f_count[0] += 1

    # Текущая точка для продолжения
    x_current = x2.copy()
    g_current = g2.copy()

    # Основной цикл Партан-метода
    for iteration in range(max_iter):
        # Проверка сходимости
        if np.linalg.norm(g_current) < tol:
            break

        # Запоминаем точки для ускоряющего шага
        x_prev = x_current.copy()
        x_start = history[-3].copy() if len(history) >= 3 else x0.copy()

        # Ускоряющий шаг одномерная оптимизация вдоль (x_prev - x_start)
        d_accel = x_prev - x_start
        if np.linalg.norm(d_accel) > 1e-12:
            alpha_accel = line_search_golden(f, x_start, d_accel, f_count)
            x_accel = x_start + alpha_accel * d_accel
            history.append(x_accel.copy())
            function_values.append(f(x_accel))
            f_count[0] += 1

            # Вычисляем градиент в ускоренной точке
            g_accel = grad(x_accel)
            g_count[0] += 1
            x_current = x_accel.copy()
            g_current = g_accel.copy()
        else:
            x_accel = x_prev
            history.append(x_accel.copy())

        # Шаг наискорейшего спуска из ускоренной точки
        d_sd = -g_current
        alpha_sd = line_search_golden(f, x_current, d_sd, f_count)
        x_sd = x_current + alpha_sd * d_sd
        history.append(x_sd.copy())

        # Вычисляем градиент в новой точке
        g_sd = grad(x_sd)
        g_count[0] += 1
        function_values.append(f(x_sd))
        f_count[0] += 1

        # Обновляем текущую точку для следующей итерации
        x_current = x_sd.copy()
        g_current = g_sd.copy()

        # Проверка сходимости
        if np.linalg.norm(g_current) < tol:
            break

    end_time = time.time()

    # Финальная точка
    x_opt = x_current
    f_opt = function_values[-1]

    return {
        'x_opt': x_opt,
        'f_opt': f_opt,
        'iterations': iteration + 1,
        'f_count': f_count[0],
        'g_count': g_count[0],
        'time': end_time - start_time,
        'converged': np.linalg.norm(grad(x_opt)) < tol,
        'final_gradient_norm': np.linalg.norm(grad(x_opt)),
        'history': np.array(history),
        'function_values': function_values
    }


# Построение графика
def plot_partan_trajectory(results: List[Dict], test_points: List[np.ndarray]):
    # Сетка
    x1 = np.linspace(-3, 7, 300)
    x2 = np.linspace(-4, 6, 300)
    X1, X2 = np.meshgrid(x1, x2)

    # Значения функции
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = rosenbrock(np.array([X1[i, j], X2[i, j]]))

    plt.figure(figsize=(14, 10))

    # Контуры с логарифмическими уровнями
    levels = np.logspace(-2, 4, 30)
    contour = plt.contour(X1, X2, Z, levels=levels, cmap='viridis', alpha=0.7)
    plt.colorbar(contour, label='f(x) - лог. шкала')

    # Цвета для разных траекторий
    colors = ['red', 'blue', 'green', 'orange']

    # Рисуем траектории для всех начальных точек
    for i, (result, point, color) in enumerate(zip(results, test_points, colors)):
        history = result['history']

        # Траектория метода
        plt.plot(history[:, 0], history[:, 1], color=color, linewidth=2,
                 marker='o', markersize=3, markevery=max(1, len(history) // 8),
                 label=f'Точка {i + 1}: ({point[0]:.3f}, {point[1]:.3f})')

        # Начальная точка (звезда)
        plt.plot(history[0, 0], history[0, 1], color=color, marker='*',
                 markersize=20, markeredgecolor='black', markeredgewidth=1.5)

        # Конечная точка (квадрат)
        plt.plot(history[-1, 0], history[-1, 1], color=color, marker='s',
                 markersize=15, markeredgecolor='black', markeredgewidth=1.5)

    # Глобальный минимум
    plt.plot(1, 1, 'y*', markersize=25, label='Глобальный минимум (1, 1)',
             markeredgecolor='black', markeredgewidth=2)

    plt.xlabel('x₁', fontsize=12)
    plt.ylabel('x₂', fontsize=12)
    plt.title('Модифицированный Партан-метод (PARTAN) на функции Розенброка', fontsize=14)
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.xlim(-3, 7)
    plt.ylim(-4, 6)
    plt.tight_layout()
    plt.show()

def main():

    # Начальные точки
    test_points = [
        np.array([1.200, 1.000]),
        np.array([-2.000, -2.000]),
        np.array([5.621, -3.635]),
        np.array([-0.221, 0.639])
    ]

    results = []

    # Запускаем оптимизацию для каждой начальной точки
    for i, x0 in enumerate(test_points):
        print(f"\n--- Тест {i + 1}: Начальная точка ({x0[0]:.3f}, {x0[1]:.3f}) ---")

        stats = partan_optimize(rosenbrock, rosenbrock_gradient, x0,
                                max_iter=500, tol=1e-8)

        results.append(stats)

        # Выводим результаты
        print(f"  Найденный минимум: ({stats['x_opt'][0]:.8f}, {stats['x_opt'][1]:.8f})")
        print(f"  Значение функции: {stats['f_opt']:.2e}")
        print(f"  Итераций PARTAN: {stats['iterations']}")
        print(f"  Вычислений функции: {stats['f_count']}")
        print(f"  Вычислений градиента: {stats['g_count']}")
        print(f"  Время выполнения: {stats['time']:.4f} с")
        print(f"  Норма градиента: {stats['final_gradient_norm']:.2e}")
        print(f"  Статус: {'СОШЁЛСЯ' if stats['converged'] else 'НЕ СОШЁЛСЯ'}")

    # Создаем сводную таблицу
    print("\n" + "=" * 90)
    print("СВОДНАЯ ТАБЛИЦА - ПАРТАН-МЕТОД")
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
            '||∇f||': f"{stats['final_gradient_norm']:.2e}",
            'Сходимость': 'Да' if stats['converged'] else 'Нет',
            '(1,1)?': 'Да' if np.allclose(stats['x_opt'], [1, 1], atol=1e-4) else 'Нет'
        })

    df = pd.DataFrame(table_data)
    print(df.to_string(index=False))

    plot_partan_trajectory(results, test_points)

    # Анализ результатов
    print("\n" + "=" * 90)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("=" * 90)

    for i, (point, stats) in enumerate(zip(test_points, results)):
        status = "✓" if stats['converged'] and np.allclose(stats['x_opt'], [1, 1], atol=1e-4) else "✗"
        dest = "глобальный минимум (1,1)" if np.allclose(stats['x_opt'], [1, 1],
                                                         atol=1e-4) else f"локальный минимум ({stats['x_opt'][0]:.3f}, {stats['x_opt'][1]:.3f})"
        print(f"{status} Точка {i + 1} ({point[0]:.3f}, {point[1]:.3f}) -> {dest}")

if __name__ == "__main__":
    # Для воспроизводимости результатов
    np.random.seed(42)
    main()