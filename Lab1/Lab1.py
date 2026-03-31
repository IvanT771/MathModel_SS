# Практическая работа 1
# Решение уравнения Лапласа методом конечных разностей.

# d2U/dx2 + d2U/dy2 = 0,   0 <= x <= 1, 0 <= y <= 1

# Граничные условия:
#U(0, y) = sin(pi * y)
#U(1, y) = 0
#U(x, 0) = 0
#U(x, 1) = 0

import math
import os
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

SHOW_PLOTS = os.environ.get("LAB1_SHOW_PLOTS", "0") == "1"
if not SHOW_PLOTS:
    matplotlib.use("Agg")

NX = 20
NY = 20

# Геометрические границы области
X_MIN = 0.0
X_MAX = 1.0
Y_MIN = 0.0
Y_MAX = 1.0
PLOTS_DIR = Path(__file__).resolve().parent / "plots"

def build_grid(nx: int, ny: int):
    """
    Создает одномерные массивы координат x и y,
    а также шаги сетки hx и hy.

    nx, ny - число интервалов по x и y.
    Количество узлов равно nx+1 и ny+1 соответственно.
    """
    x = np.linspace(X_MIN, X_MAX, nx + 1)
    y = np.linspace(Y_MIN, Y_MAX, ny + 1)

    hx = (X_MAX - X_MIN) / nx
    hy = (Y_MAX - Y_MIN) / ny

    return x, y, hx, hy


# Граничные условия
def apply_boundary_conditions(u: np.ndarray, x: np.ndarray, y: np.ndarray):
    # Нижняя граница: y = 0
    u[0, :] = 0.0
    # Верхняя граница: y = 1
    u[-1, :] = 0.0
    # Правая граница: x = 1
    u[:, -1] = 0.0
    # Левая граница: x = 0
    u[:, 0] = np.sin(np.pi * y)


def create_initial_solution(nx: int, ny: int):
    x, y, hx, hy = build_grid(nx, ny)

    # Матрица размером (ny+1) строк на (nx+1) столбцов
    u = np.zeros((ny + 1, nx + 1), dtype=float)

    apply_boundary_conditions(u, x, y)
    return x, y, hx, hy, u


# Адаптированная тестовая задача для верификации (1D профиль)
def apply_test_boundary_conditions(u: np.ndarray, u_left: float, u_right: float):
    """
    Граничные условия тестовой задачи:
    - слева/справа фиксированные значения (Дирихле),
    - сверху/снизу "непротекание" (дублирование соседнего внутреннего слоя).
    """
    # x = 0 и x = 1
    u[:, 0] = u_left
    u[:, -1] = u_right

    # y = 0 и y = 1 (условие отсутствия перетекания по y)
    u[0, 1:-1] = u[1, 1:-1]
    u[-1, 1:-1] = u[-2, 1:-1]


def create_test_initial_solution(nx: int, ny: int, u_left: float, u_right: float):
    x, y, hx, hy = build_grid(nx, ny)
    u = np.zeros((ny + 1, nx + 1), dtype=float)
    apply_test_boundary_conditions(u, u_left, u_right)
    return x, y, hx, hy, u


def libman_fixed_iterations_test_task(
    nx: int,
    ny: int,
    iterations: int,
    u_left: float = 100.0,
    u_right: float = 40.0
):
    """
    Верификационная тестовая задача:
    решение после заданного числа итераций с адаптированными границами.
    """
    x, y, hx, hy, u = create_test_initial_solution(nx, ny, u_left, u_right)

    for _ in range(iterations):
        for i in range(1, ny):
            for j in range(1, nx):
                u[i, j] = (
                    hy * hy * (u[i, j + 1] + u[i, j - 1]) +
                    hx * hx * (u[i + 1, j] + u[i - 1, j])
                ) / (2.0 * (hx * hx + hy * hy))

        apply_test_boundary_conditions(u, u_left, u_right)

    return x, y, u


def analytical_solution_test_task(
    x: np.ndarray,
    y: np.ndarray,
    u_left: float = 100.0,
    u_right: float = 40.0
):
    """
    Аналитическое решение тестовой задачи: линейный профиль по x,
    одинаковый для всех y.
    """
    line = u_left + (u_right - u_left) * (x - X_MIN) / (X_MAX - X_MIN)
    return np.tile(line, (len(y), 1))


# Аналитическое решение исходной 2D задачи
def analytical_solution(x: np.ndarray, y: np.ndarray):
    """
    Аналитическое решение для задачи:
        U_xx + U_yy = 0
        U(0, y) = sin(pi*y)
        U(1, y) = 0
        U(x, 0) = 0
        U(x, 1) = 0

    Решение:
        U(x, y) = sinh(pi * (1 - x)) / sinh(pi) * sin(pi * y)
    """
    result = np.zeros((len(y), len(x)), dtype=float)

    for i in range(len(y)):
        for j in range(len(x)):
            result[i, j] = (
                math.sinh(math.pi * (1.0 - x[j])) / math.sinh(math.pi)
            ) * math.sin(math.pi * y[i])

    return result


# Метод Либмана для заданного числа итераций
def libman_fixed_iterations(nx: int, ny: int, iterations: int):
    """
    Решение методом Либмана для заданного числа итераций.

    Здесь используется итерационная формула для внутренних узлов.
    Для случая hx = hy формула сводится к среднему четырех соседей.

    Для общего случая:
        u_new(i,j) =
            [ hy^2 * (u(i,j+1) + u(i,j-1)) + hx^2 * (u(i+1,j) + u(i-1,j)) ]
            / [ 2 * (hx^2 + hy^2) ]

    """
    x, y, hx, hy, u = create_initial_solution(nx, ny)

    for _ in range(iterations):
        # Проходим только по внутренним узлам
        for i in range(1, ny):
            for j in range(1, nx):
                u[i, j] = (
                    hy * hy * (u[i, j + 1] + u[i, j - 1]) +
                    hx * hx * (u[i + 1, j] + u[i - 1, j])
                ) / (2.0 * (hx * hx + hy * hy))

    return x, y, u


# Метод Либмана до заданной точности
def libman_until_eps(nx: int, ny: int, eps: float, max_iterations: int = 100000):
    """
    Решение методом Либмана до достижения точности eps.

    Критерий остановки:
        max |u_new - u_old| < eps
    """
    x, y, hx, hy, u = create_initial_solution(nx, ny)

    iteration_count = 0

    while iteration_count < max_iterations:
        max_diff = 0.0

        # Обновляем внутренние узлы
        for i in range(1, ny):
            for j in range(1, nx):
                old_value = u[i, j]

                new_value = (
                    hy * hy * (u[i, j + 1] + u[i, j - 1]) +
                    hx * hx * (u[i + 1, j] + u[i - 1, j])
                ) / (2.0 * (hx * hx + hy * hy))

                u[i, j] = new_value

                diff = abs(new_value - old_value)
                if diff > max_diff:
                    max_diff = diff

        iteration_count += 1

        if max_diff < eps:
            return x, y, u, iteration_count, max_diff

    return x, y, u, iteration_count, max_diff


# Ускоренный метод Либмана (релаксация)
def libman_relaxation_until_eps(
    nx: int,
    ny: int,
    eps: float,
    w: float,
    max_iterations: int = 100000
):
    """
    Ускоренный метод Либмана с коэффициентом релаксации w.

    Формула релаксации:
        u_new = w * u_libman + (1 - w) * u_old

    где:
    - u_libman - значение, вычисленное обычной формулой Либмана,
    - u_old    - старое значение узла,
    - w        - коэффициент релаксации.

    """
    x, y, hx, hy, u = create_initial_solution(nx, ny)

    iteration_count = 0

    while iteration_count < max_iterations:
        max_diff = 0.0

        for i in range(1, ny):
            for j in range(1, nx):
                old_value = u[i, j]

                libman_value = (
                    hy * hy * (u[i, j + 1] + u[i, j - 1]) +
                    hx * hx * (u[i + 1, j] + u[i - 1, j])
                ) / (2.0 * (hx * hx + hy * hy))

                # Релаксационная поправка
                new_value = w * libman_value + (1.0 - w) * old_value

                u[i, j] = new_value

                diff = abs(new_value - old_value)
                if diff > max_diff:
                    max_diff = diff

        iteration_count += 1

        if max_diff < eps:
            return x, y, u, iteration_count, max_diff

    return x, y, u, iteration_count, max_diff



# Верификация на аналитическом решении

def compute_error(u_numeric: np.ndarray, u_exact: np.ndarray):
    """
    Возвращает максимальную абсолютную ошибку между
    численным и аналитическим решениями.
    """
    return np.max(np.abs(u_numeric - u_exact))


# Исследование зависимости числа итераций от eps

def study_eps_dependency(nx: int, ny: int, eps_values):
    """
    Для каждого eps решает задачу методом Либмана
    и сохраняет число итераций до сходимости.
    """
    iterations = []

    for eps in eps_values:
        _, _, _, iters, _ = libman_until_eps(nx, ny, eps)
        iterations.append(iters)

    return iterations


# Исследование зависимости числа итераций от w
def study_w_dependency(nx: int, ny: int, eps: float, w_values):
    """
    Для фиксированной точности eps исследует,
    как меняется число итераций при разных w.
    """
    iterations = []

    for w in w_values:
        _, _, _, iters, _ = libman_relaxation_until_eps(nx, ny, eps, w)
        iterations.append(iters)

    return iterations



# Печать результатов
def print_matrix(title: str, matrix: np.ndarray, precision: int = 6):
    print(f"\n{title}")
    print(np.array2string(matrix, precision=precision, suppress_small=False))


# Графики
def plot_eps_dependency(eps_values, iterations):
    """
    Строит график зависимости числа итераций от точности eps.
    """
    sorted_pairs = sorted(zip(eps_values, iterations))
    sorted_eps = [eps for eps, _ in sorted_pairs]
    sorted_iterations = [iteration for _, iteration in sorted_pairs]

    plt.figure(figsize=(8, 5))
    plt.plot(sorted_eps, sorted_iterations, marker='o', linestyle='--')
    plt.xscale('log')
    plt.xlabel('Точность eps')
    plt.ylabel('Число итераций')
    plt.title('Зависимость числа итераций от точности (метод Либмана)')
    plt.grid(True)
    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PLOTS_DIR / "eps_dependency.png"
    plt.savefig(output_path, dpi=180)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()
    return output_path


def plot_w_dependency(w_values, iterations, baseline_iterations=None):
    """
    Строит график зависимости числа итераций от коэффициента релаксации w.
    При baseline_iterations можно показать горизонтальную лиeнию для обычного метода
    """
    plt.figure(figsize=(8, 5))
    plt.plot(w_values, iterations, marker='o', label='Ускоренный метод Либмана')

    if baseline_iterations is not None:
        plt.axhline(
            y=baseline_iterations,
            linestyle='--',
            label='Обычный метод Либмана'
        )

    plt.xlabel('Коэффициент релаксации w')
    plt.ylabel('Число итераций')
    plt.title('Зависимость числа итераций от коэффициента релаксации')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PLOTS_DIR / "w_dependency.png"
    plt.savefig(output_path, dpi=180)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()
    return output_path


def plot_solution_comparison(x, y, u_numeric: np.ndarray):
    """
    Строит объемный график численного решения на сетке.
    """
    x_grid, y_grid = np.meshgrid(x, y)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        x_grid,
        y_grid,
        u_numeric,
        cmap="viridis",
        edgecolor="black",
        linewidth=0.4,
        antialiased=True
    )
    fig.colorbar(surface, ax=ax, shrink=0.7, pad=0.12, label="U(x, y)")
    ax.set_title("решение уравнения Лапласа")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("U(x, y)")
    ax.set_xticks(np.arange(X_MIN, X_MAX + 0.001, 0.1))
    ax.set_yticks(np.arange(Y_MIN, Y_MAX + 0.001, 0.1))
    ax.view_init(elev=28, azim=-135)

    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PLOTS_DIR / "solution_comparison.png"
    plt.savefig(output_path, dpi=180)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()
    return output_path

def main():
    # 1. Решение методом Либмана для заданного числа итераций
    fixed_iterations = 50
    x, y, u_fixed = libman_fixed_iterations(NX, NY, fixed_iterations)

    print_matrix(
        f"Численное решение методом Либмана после {fixed_iterations} итераций:",
        u_fixed
    )


    #1.1 Верификация на адаптированной тестовой задаче
    x_test, y_test, u_test = libman_fixed_iterations_test_task(
        NX, NY, fixed_iterations, u_left=100.0, u_right=40.0
    )

    u_test_exact = analytical_solution_test_task(x_test, y_test, 100.0, 40.0)
    test_error = compute_error(u_test, u_test_exact)

    print_matrix(
        "Верификационная тестовая задача (адаптированные границы), численное решение:",
        u_test
    )

    print_matrix(
        "Верификационная тестовая задача, аналитическое решение:",
        u_test_exact
    )

    print(f"Максимальная ошибка тестовой задачи: {test_error:.10f}")

    #2. Аналитическое решение и сравнение
    u_exact = analytical_solution(x, y)

    print_matrix("Аналитическое решение на той же сетке:", u_exact)

    max_error_fixed = compute_error(u_fixed, u_exact)
    print(f"\nМаксимальная ошибка после {fixed_iterations} итераций: {max_error_fixed:.10f}")
    solution_plot_path = plot_solution_comparison(x, y, u_fixed)
    print(f"График решения сохранен в: {solution_plot_path}")


    #3.Решение до заданной точности eps
    eps = 1e-4
    x_eps, y_eps, u_eps, iter_eps, last_diff = libman_until_eps(NX, NY, eps)

    print_matrix(
        f"Решение методом Либмана до точности eps={eps}:",
        u_eps
    )
    print(f"\nМетод Либмана сошелся за {iter_eps} итераций")
    print(f"Последнее max |u_new - u_old| = {last_diff:.10e}")

    u_exact_eps = analytical_solution(x_eps, y_eps)
    max_error_eps = compute_error(u_eps, u_exact_eps)
    print(f"Максимальная ошибка относительно аналитического решения: {max_error_eps:.10f}")


    # 4. Исследование зависимости числа итераций от eps
    eps_values = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    eps_iterations = study_eps_dependency(NX, NY, eps_values)

    print("\nЗависимость числа итераций от eps:")
    for eps_value, iters in zip(eps_values, eps_iterations):
        print(f"eps = {eps_value:.0e} -> итераций = {iters}")

    eps_plot_path = plot_eps_dependency(eps_values, eps_iterations)
    print(f"График зависимости от eps сохранен в: {eps_plot_path}")

    # 5. Ускоренный метод Либмана
    eps_relax = 1e-7
    w_values = np.arange(1.0, 1.71, 0.05)

    relax_iterations = study_w_dependency(NX, NY, eps_relax, w_values)

    _, _, _, baseline_iters, _ = libman_until_eps(NX, NY, eps_relax)

    print("\nЗависимость числа итераций от коэффициента релаксации w:")
    for w, iters in zip(w_values, relax_iterations):
        print(f"w = {w:.2f} -> итераций = {iters}")

    w_plot_path = plot_w_dependency(w_values, relax_iterations, baseline_iters)
    print(f"График зависимости от w сохранен в: {w_plot_path}")

    # 6. Пример ускоренного метода для одного конкретного w
    chosen_w = 1.2
    x_w, y_w, u_w, iter_w, diff_w = libman_relaxation_until_eps(
        NX, NY, eps_relax, chosen_w
    )

    print_matrix(
        f"Решение ускоренным методом Либмана при w={chosen_w}, eps={eps_relax}:",
        u_w
    )
    print(f"\nУскоренный метод сошелся за {iter_w} итераций")
    print(f"Последнее max |u_new - u_old| = {diff_w:.10e}")

    u_exact_w = analytical_solution(x_w, y_w)
    max_error_w = compute_error(u_w, u_exact_w)
    print(f"Максимальная ошибка ускоренного метода: {max_error_w:.10f}")


if __name__ == "__main__":
    main()
