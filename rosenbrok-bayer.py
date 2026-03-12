import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Callable, List, Dict, Tuple, Optional
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


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


# Локальная оптимизация
def local_optimization(f: Callable, x0: np.ndarray, bounds: List[Tuple[float, float]],
                       grad: Optional[Callable] = None) -> Tuple[np.ndarray, float, Dict]:
    """
    Локальная оптимизация с использованием.
    Возвращает точку минимума, значение функции и статистику.
    """
    start_time = time.time()

    result = minimize(
        f, x0,
        method='L-BFGS-B',
        bounds=bounds,
        jac=grad,
        options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8}
    )

    stats = {
        'f_count': result.nfev,
        'g_count': result.njev if hasattr(result, 'njev') else 0,
        'iterations': result.nit,
        'success': result.success,
        'time': time.time() - start_time
    }

    return result.x, result.fun, stats


# Квадратичная интерполяционная модель
class QuadraticInterpolationModel:
    """
    Квадратичная интерполяционная модель для метода Байера.
    f(x) ≈ c + g^T x + 0.5 * x^T H x
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.n_coeffs = (dim + 1) * (dim + 2) // 2  # число коэффициентов
        self.coeffs = None
        self.H = None  # матрица Гессе
        self.g = None  # градиент
        self.c = None  # константа

    def fit(self, points: np.ndarray, values: np.ndarray) -> bool:
        """
        Строит интерполяционную модель по точкам.
        points: массив точек (n_points x dim)
        values: значения функции в точках (n_points)
        """
        n_points = len(points)

        if n_points < self.n_coeffs:
            return False

        # Строим матрицу для МНК
        A = np.zeros((n_points, self.n_coeffs))

        for i in range(n_points):
            x = points[i]
            col = 0

            # Линейные члены
            for j in range(self.dim):
                A[i, col] = x[j]
                col += 1

            # Квадратичные члены (x_j * x_k для j <= k)
            for j in range(self.dim):
                for k in range(j, self.dim):
                    A[i, col] = x[j] * x[k]
                    col += 1

            # Свободный член
            A[i, -1] = 1.0

        # Решаем МНК
        try:
            self.coeffs, _, _, _ = np.linalg.lstsq(A, values, rcond=None)

            # Восстанавливаем матрицу Гессе и градиент
            self._extract_hessian_gradient()
            return True
        except:
            return False

    # Извлекаем матрицу Гессе и градиент из функции
    def _extract_hessian_gradient(self):
        self.H = np.zeros((self.dim, self.dim))
        self.g = np.zeros(self.dim)

        col = 0

        # Линейные члены
        for j in range(self.dim):
            self.g[j] = self.coeffs[col]
            col += 1

        # Квадратичные члены
        for j in range(self.dim):
            for k in range(j, self.dim):
                self.H[j, k] = self.coeffs[col]
                self.H[k, j] = self.coeffs[col]
                col += 1

        # Свободный член
        self.c = self.coeffs[-1]

    # Предсказываем значение в точке х
    def predict(self, x: np.ndarray) -> float:
        if self.coeffs is None:
            return np.inf

        return self.c + self.g @ x + 0.5 * x @ self.H @ x

    # Находим минимум квадратичной модели в заданных границах
    def get_minimum(self, bounds: List[Tuple[float, float]]) -> np.ndarray:
        if self.coeffs is None:
            # Если модели нет, возвращаем случайную точку
            return np.random.uniform(
                [b[0] for b in bounds],
                [b[1] for b in bounds]
            )

        # Проверяем на выпуклость
        eigenvalues = np.linalg.eigvals(self.H)

        if np.min(eigenvalues) > 1e-10:
            # Модель выпукла - есть аналитическое решение
            try:
                x_min = -np.linalg.solve(self.H, self.g)
                # Ограничиваем границами
                x_min = np.clip(x_min, [b[0] for b in bounds], [b[1] for b in bounds])
                return x_min
            except:
                pass

        # Модель не выпукла или сингулярна - используем несколько случайных попыток
        best_x = None
        best_f = np.inf

        for _ in range(20):
            x_candidate = np.random.uniform(
                [b[0] for b in bounds],
                [b[1] for b in bounds]
            )
            f_candidate = self.predict(x_candidate)

            if f_candidate < best_f:
                best_f = f_candidate
                best_x = x_candidate

        return best_x if best_x is not None else np.array([(b[0] + b[1]) / 2 for b in bounds])


# Функции для работы с множеством локальных минимумов
def cluster_minima(minima: List[Tuple[np.ndarray, float]], tol: float = 1e-2) -> List[Tuple[np.ndarray, float]]:
    if len(minima) <= 1:
        return minima

    points = np.array([p for p, _ in minima])
    values = np.array([v for _, v in minima])

    # Вычисляем попарные расстояния
    distances = cdist(points, points)

    # Кластеризация
    used = np.zeros(len(points), dtype=bool)
    unique_minima = []

    for i in range(len(points)):
        if used[i]:
            continue

        # Находим все точки в кластере
        cluster_indices = np.where(distances[i] < tol)[0]
        cluster_indices = [idx for idx in cluster_indices if not used[idx]]

        if not cluster_indices:
            continue

        # Выбираем точку с минимальным значением функции
        best_in_cluster = min(cluster_indices, key=lambda idx: values[idx])
        unique_minima.append((points[best_in_cluster], values[best_in_cluster]))

        # Помечаем все точки кластера как использованные
        for idx in cluster_indices:
            used[idx] = True

    return unique_minima


def add_minimum(minima: List[Tuple[np.ndarray, float]],
                new_point: np.ndarray,
                new_value: float,
                max_size: int = 20) -> List[Tuple[np.ndarray, float]]:

    # Добавляем новый минимум в список, поддерживая максимальный размер
    minima.append((new_point.copy(), new_value))

    # Кластеризуем
    minima = cluster_minima(minima)

    # Сортируем по значению функции
    minima.sort(key=lambda x: x[1])

    # Ограничиваем размер
    if len(minima) > max_size:
        minima = minima[:max_size]

    return minima


# Основная реализация метода Байера
def bauer_global_optimization(f: Callable,
                              bounds: List[Tuple[float, float]],
                              grad: Optional[Callable] = None,
                              n_initial_points: int = 20,
                              n_interpolation_points: int = 8,
                              max_iterations: int = 30,
                              local_opt_tol: float = 1e-6) -> Dict:
    """
    Полная реализация метода Байера для глобальной оптимизации.

    Алгоритм:
    1. Генерирует начальные точки и находит локальные минимумы
    2. Строит квадратичную интерполяцию между лучшими минимумами
    3. Использует интерполяцию для предсказания новых перспективных областей
    4. Повторяет, уточняя область поиска
    """
    start_time = time.time()
    dim = len(bounds)

    # Статистика
    f_count = 0
    g_count = 0
    history = []  # история всех исследованных точек
    local_minima = []  # найденные локальные минимумы

    # Этап 1 Начальное исследование
    initial_points = np.random.uniform(
        [b[0] for b in bounds],
        [b[1] for b in bounds],
        size=(n_initial_points, dim)
    )

    # Добавляем угловые точки для лучшего покрытия
    corners = []
    for i in range(2 ** dim):
        corner = []
        for j in range(dim):
            corner.append(bounds[j][i >> j & 1])
        corners.append(np.array(corner))

    initial_points = np.vstack([initial_points] + corners)

    # Находим локальные минимумы из каждой начальной точки
    for i, x0 in enumerate(initial_points):
        x_min, f_min, stats = local_optimization(f, x0, bounds, grad)
        f_count += stats['f_count']
        g_count += stats['g_count']

        history.append({
            'type': 'local_min',
            'start': x0.copy(),
            'x': x_min.copy(),
            'f': f_min,
            'iteration': 0
        })

        local_minima = add_minimum(local_minima, x_min, f_min)

    # Этап 2 Итеративное уточнение через интерполяцию
    current_bounds = [list(b) for b in bounds]  # копируем границы

    for iteration in range(max_iterations):
        # Выбираем точки для интерполяции (лучшие + разнообразные)
        n_interp = min(n_interpolation_points, len(local_minima))
        best_points = [p for p, _ in local_minima[:n_interp]]
        best_values = [v for _, v in local_minima[:n_interp]]

        # Добавляем несколько случайных точек для разнообразия
        if len(best_points) < n_interpolation_points:
            n_random = n_interpolation_points - len(best_points)
            random_points = np.random.uniform(
                [b[0] for b in current_bounds],
                [b[1] for b in current_bounds],
                size=(n_random, dim)
            )
            best_points.extend(random_points)
            for rp in random_points:
                best_values.append(f(rp))
                f_count += 1

        # Строим интерполяционную модель
        model = QuadraticInterpolationModel(dim)
        if not model.fit(np.array(best_points), np.array(best_values)):
            # Сужаем область поиска вокруг лучшей точки
            best_point = local_minima[0][0]
            for j in range(dim):
                width = (current_bounds[j][1] - current_bounds[j][0]) * 0.8
                current_bounds[j][0] = max(bounds[j][0], best_point[j] - width / 2)
                current_bounds[j][1] = min(bounds[j][1], best_point[j] + width / 2)
            continue

        # Находим минимум модели
        x_pred = model.get_minimum(current_bounds)

        # Исследуем предсказанную точку
        x_min, f_min, stats = local_optimization(f, x_pred, bounds, grad)
        f_count += stats['f_count']
        g_count += stats['g_count']

        history.append({
            'type': 'predicted',
            'prediction': x_pred.copy(),
            'x': x_min.copy(),
            'f': f_min,
            'iteration': iteration + 1
        })

        # Добавляем новый минимум
        old_best = local_minima[0][1]
        local_minima = add_minimum(local_minima, x_min, f_min)
        new_best = local_minima[0][1]

        if f_min < old_best - 1e-6:
            print(f"    Улучшение! Новый лучший минимум: f={new_best:.2e}")

        # Сужаем область поиска вокруг лучшей точки
        if iteration < max_iterations - 1:
            best_point = local_minima[0][0]
            shrink_factor = 0.85

            for j in range(dim):
                width = (current_bounds[j][1] - current_bounds[j][0]) * shrink_factor
                new_center = best_point[j]
                current_bounds[j][0] = max(bounds[j][0], new_center - width / 2)
                current_bounds[j][1] = min(bounds[j][1], new_center + width / 2)

        # Проверка сходимости
        if len(local_minima) >= 3:
            values = [v for _, v in local_minima[:3]]
            if max(values) - min(values) < local_opt_tol:
                break

    end_time = time.time()

    # Финальный результат
    x_opt, f_opt = local_minima[0]

    # Дополнительная локальная оптимизация для гарантии
    x_opt, f_opt, final_stats = local_optimization(f, x_opt, bounds, grad)
    f_count += final_stats['f_count']
    g_count += final_stats['g_count']

    return {
        'x_opt': x_opt,
        'f_opt': f_opt,
        'iterations': iteration + 1,
        'f_count': f_count,
        'g_count': g_count,
        'time': end_time - start_time,
        'n_local_minima': len(local_minima),
        'all_minima': [(p.copy(), v) for p, v in local_minima],
        'history': history,
        'converged': True
    }


# Визуализация
def plot_bauer_trajectory(results: List[Dict], test_points: List[np.ndarray]):
    # Сетка
    x1 = np.linspace(-3, 7, 300)
    x2 = np.linspace(-4, 6, 300)
    X1, X2 = np.meshgrid(x1, x2)

    # Значения функции
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = rosenbrock(np.array([X1[i, j], X2[i, j]]))

    plt.figure(figsize=(16, 12))

    # Контуры
    levels = np.logspace(-2, 4, 30)
    contour = plt.contour(X1, X2, Z, levels=levels, cmap='viridis', alpha=0.6)
    plt.colorbar(contour, label='f(x) - лог. шкала')

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']

    for i, (result, point, color) in enumerate(zip(results, test_points, colors)):
        history = result['history']

        # Начальная точка
        plt.plot(point[0], point[1], color=color, marker='*',
                 markersize=20, markeredgecolor='black', markeredgewidth=1.5,
                 label=f'Начальная точка {i + 1}')

        # Все найденные локальные минимумы
        minima_points = [item['x'] for item in history if item['type'] == 'local_min']
        if minima_points:
            minima_points = np.array(minima_points)
            plt.scatter(minima_points[:, 0], minima_points[:, 1],
                        c=color, s=50, alpha=0.6, marker='o',
                        edgecolors='black', linewidths=1)

        # Предсказанные точки
        pred_points = [item['prediction'] for item in history if item['type'] == 'predicted']
        if pred_points:
            pred_points = np.array(pred_points)
            plt.scatter(pred_points[:, 0], pred_points[:, 1],
                        c=color, s=80, alpha=0.8, marker='^',
                        edgecolors='black', linewidths=1)

        # Финальный минимум для этой траектории
        final_min = result['x_opt']
        plt.plot(final_min[0], final_min[1], color=color, marker='s',
                 markersize=15, markeredgecolor='black', markeredgewidth=2)

    # Глобальный минимум
    plt.plot(1, 1, 'y*', markersize=25, label='Глобальный минимум (1, 1)',
             markeredgecolor='black', markeredgewidth=2)

    plt.xlabel('x1', fontsize=12)
    plt.ylabel('x2', fontsize=12)
    plt.title('Метод Байера: глобальная оптимизация через интерполяцию между локальными минимумами', fontsize=14)
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.xlim(-3, 7)
    plt.ylim(-4, 6)
    plt.tight_layout()
    plt.show()


# Основная функция тестирования
def main_bauer():

    test_points = [
        np.array([1.200, 1.000]),
        np.array([-2.000, -2.000]),
        np.array([5.621, -3.635]),
        np.array([-0.221, 0.639])
    ]

    bounds = [(-3.0, 7.0), (-4.0, 6.0)]

    results = []

    for i, x0 in enumerate(test_points):
        print(f"\n--- Тест {i + 1}: Начальная точка ({x0[0]:.3f}, {x0[1]:.3f}) ---")

        # Запускаем метод Байера
        stats = bauer_global_optimization(
            rosenbrock, bounds,
            grad=rosenbrock_gradient,
            n_initial_points=15,
            n_interpolation_points=8,
            max_iterations=15
        )
        results.append(stats)

        print(f"\n  РЕЗУЛЬТАТ:")
        print(f"    Найденный минимум: ({stats['x_opt'][0]:.8f}, {stats['x_opt'][1]:.8f})")
        print(f"    f(x*) = {stats['f_opt']:.2e}")
        print(f"    Итераций (внешних): {stats['iterations']}")
        print(f"    Вызовов f(x): {stats['f_count']}")
        print(f"    Вызовов ∇f: {stats['g_count']}")
        print(f"    Время: {stats['time']:.4f} с")
        print(f"    Найдено локальных минимумов: {stats['n_local_minima']}")
        print(f"    Статус: {'ДОСТИГ (1,1)' if np.allclose(stats['x_opt'], [1, 1], atol=1e-4) else 'НЕ ДОСТИГ'}")

        # Показываем все найденные минимумы
        if stats['n_local_minima'] > 1:
            print(f"    Все найденные минимумы:")
            for j, (p, v) in enumerate(stats['all_minima'][:5]):  # показываем первые 5
                print(f"      {j + 1}: ({p[0]:.6f}, {p[1]:.6f}), f={v:.2e}")

    # Таблица результатов
    print("\n" + "=" * 90)
    print("СВОДНАЯ ТАБЛИЦА")
    print("=" * 90)

    table_data = []
    for i, (point, stats) in enumerate(zip(test_points, results)):
        table_data.append({
            '№': i + 1,
            'Начальная точка': f"({point[0]:.3f}, {point[1]:.3f})",
            'Найденный минимум': f"({stats['x_opt'][0]:.6f}, {stats['x_opt'][1]:.6f})",
            'f(x*)': f"{stats['f_opt']:.2e}",
            'Итерации': stats['iterations'],
            'Вызовы f': stats['f_count'],
            'Лок. минимумы': stats['n_local_minima'],
            'Время (с)': f"{stats['time']:.4f}",
            '(1,1)?': 'Да' if np.allclose(stats['x_opt'], [1, 1], atol=1e-4) else 'Нет'
        })

    df = pd.DataFrame(table_data)
    print(df.to_string(index=False))

    plot_bauer_trajectory(results, test_points)

if __name__ == "__main__":
    main_bauer()