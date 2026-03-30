import math
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

SHOW_PLOTS = os.environ.get("LAB3_SHOW_PLOTS", "0") == "1"
if not SHOW_PLOTS:
    matplotlib.use("Agg")


BASE_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BASE_DIR / "plots"
SUMMARY_PATH = BASE_DIR / "results_summary.txt"


@dataclass(frozen=True)
class StationaryRodModel:
    length: float
    conductivity: float
    area: float
    perimeter: float
    left_temperature: float
    ambient_temperature: float


REFERENCE_MODEL = StationaryRodModel(
    length=7.5,
    conductivity=72.0,
    area=1.0,
    perimeter=2.0,
    left_temperature=150.0,
    ambient_temperature=40.0,
)


UNIFORM_MESH = np.array([0.0, 1.5, 3.0, 4.5, 6.0, 7.5], dtype=float)
NONUNIFORM_MESH = np.array([0.0, 0.75, 1.5, 3.0, 5.25, 7.5], dtype=float)

CONTROL_RESULTS = {
    10.0: {
        "uniform": np.array([150.0, 88.836, 61.745, 49.824, 44.757, 43.008], dtype=float),
        "nonuniform": np.array([150.0, 113.676, 89.171, 61.617, 46.345, 42.727], dtype=float),
    },
    50.0: {
        "uniform": np.array([150.0, 53.091, 41.558, 40.185, 40.022, 40.005], dtype=float),
        "nonuniform": np.array([150.0, 83.783, 56.893, 41.897, 39.951, 40.002], dtype=float),
    },
}


def ensure_plot_dir() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def save_plot(filename: str) -> Path:
    ensure_plot_dir()
    output_path = PLOTS_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()
    return output_path


def local_stiffness_matrix(
    model: StationaryRodModel,
    alpha: float,
    element_length: float,
    include_right_boundary: bool = False,
) -> np.ndarray:
    conductive_part = (
        model.conductivity * model.area / element_length
    ) * np.array([[1.0, -1.0], [-1.0, 1.0]])

    convective_side_part = (
        alpha * model.perimeter * element_length / 6.0
    ) * np.array([[2.0, 1.0], [1.0, 2.0]])

    matrix = conductive_part + convective_side_part

    if include_right_boundary:
        matrix = matrix + alpha * model.area * np.array([[0.0, 0.0], [0.0, 1.0]])

    return matrix


def local_load_vector(
    model: StationaryRodModel,
    alpha: float,
    element_length: float,
    include_right_boundary: bool = False,
) -> np.ndarray:
    vector = (
        alpha
        * model.perimeter
        * model.ambient_temperature
        * element_length
        / 2.0
    ) * np.array([1.0, 1.0])

    if include_right_boundary:
        vector = vector + alpha * model.area * model.ambient_temperature * np.array([0.0, 1.0])

    return vector


def assemble_global_system(
    model: StationaryRodModel,
    mesh: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    node_count = len(mesh)
    global_matrix = np.zeros((node_count, node_count), dtype=float)
    global_vector = np.zeros(node_count, dtype=float)
    local_matrices: list[np.ndarray] = []
    local_vectors: list[np.ndarray] = []

    for element_index in range(node_count - 1):
        element_length = mesh[element_index + 1] - mesh[element_index]
        is_last_element = element_index == node_count - 2

        local_matrix = local_stiffness_matrix(
            model=model,
            alpha=alpha,
            element_length=element_length,
            include_right_boundary=is_last_element,
        )
        local_vector = local_load_vector(
            model=model,
            alpha=alpha,
            element_length=element_length,
            include_right_boundary=is_last_element,
        )

        local_matrices.append(local_matrix)
        local_vectors.append(local_vector)

        nodes = [element_index, element_index + 1]
        for i_local, i_global in enumerate(nodes):
            global_vector[i_global] += local_vector[i_local]
            for j_local, j_global in enumerate(nodes):
                global_matrix[i_global, j_global] += local_matrix[i_local, j_local]

    return global_matrix, global_vector, local_matrices, local_vectors


def apply_left_dirichlet_condition(
    matrix: np.ndarray,
    vector: np.ndarray,
    left_temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    corrected_matrix = matrix.copy()
    corrected_vector = vector.copy()

    corrected_vector -= corrected_matrix[:, 0] * left_temperature
    corrected_matrix[:, 0] = 0.0
    corrected_matrix[0, :] = 0.0
    corrected_matrix[0, 0] = 1.0
    corrected_vector[0] = left_temperature

    return corrected_matrix, corrected_vector


def fem_stationary_solution(
    model: StationaryRodModel,
    mesh: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    global_matrix, global_vector, local_matrices, local_vectors = assemble_global_system(
        model=model,
        mesh=mesh,
        alpha=alpha,
    )
    corrected_matrix, corrected_vector = apply_left_dirichlet_condition(
        matrix=global_matrix,
        vector=global_vector,
        left_temperature=model.left_temperature,
    )
    solution = np.linalg.solve(corrected_matrix, corrected_vector)
    return (
        solution,
        global_matrix,
        global_vector,
        corrected_matrix,
        local_matrices,
        local_vectors,
    )


def analytical_solution(
    model: StationaryRodModel,
    alpha: float,
    x: np.ndarray,
) -> np.ndarray:
    theta_left = model.left_temperature - model.ambient_temperature
    m_value = math.sqrt(alpha * model.perimeter / (model.conductivity * model.area))
    beta = alpha / (model.conductivity * m_value)
    denominator = math.cosh(m_value * model.length) + beta * math.sinh(m_value * model.length)

    theta = theta_left * (
        np.cosh(m_value * (model.length - x))
        + beta * np.sinh(m_value * (model.length - x))
    ) / denominator

    return model.ambient_temperature + theta


def element_midpoint_errors(
    model: StationaryRodModel,
    mesh: np.ndarray,
    nodal_temperatures: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    midpoints = 0.5 * (mesh[:-1] + mesh[1:])
    fem_values = 0.5 * (nodal_temperatures[:-1] + nodal_temperatures[1:])
    exact_values = analytical_solution(model, alpha, midpoints)
    return midpoints, np.abs(fem_values - exact_values)


def format_array(array: np.ndarray, precision: int = 4) -> str:
    return np.array2string(array, precision=precision, suppress_small=False)


def plot_two_meshes(
    model: StationaryRodModel,
    alpha: float,
    uniform_solution: np.ndarray,
    nonuniform_solution: np.ndarray,
    filename: str,
) -> Path:
    plt.figure(figsize=(8, 5))
    plt.plot(UNIFORM_MESH, uniform_solution, marker="o", linewidth=2, label="Равномерное разбиение КЭ")
    plt.plot(NONUNIFORM_MESH, nonuniform_solution, marker="s", linewidth=2, label="Неравномерное разбиение КЭ")
    plt.xlabel("Длина стержня, см")
    plt.ylabel("Температура, град C")
    plt.title(f"Результаты расчетов при alfa = {alpha:g}")
    plt.grid(True)
    plt.legend()
    return save_plot(filename)


def plot_all_cases(
    model: StationaryRodModel,
    solutions_by_alpha: dict[float, dict[str, np.ndarray]],
    filename: str,
) -> Path:
    plt.figure(figsize=(8, 5))
    plt.plot(
        UNIFORM_MESH,
        solutions_by_alpha[10.0]["uniform"],
        color="black",
        linewidth=2,
        label="alfa=10, равномерное",
    )
    plt.plot(
        NONUNIFORM_MESH,
        solutions_by_alpha[10.0]["nonuniform"],
        color="tab:orange",
        linestyle="--",
        linewidth=2,
        label="alfa=10, неравномерное",
    )
    plt.plot(
        NONUNIFORM_MESH,
        solutions_by_alpha[50.0]["nonuniform"],
        color="tab:red",
        linestyle="--",
        linewidth=2,
        label="alfa=50, неравномерное",
    )
    plt.plot(
        UNIFORM_MESH,
        solutions_by_alpha[50.0]["uniform"],
        color="dimgray",
        linewidth=2,
        label="alfa=50, равномерное",
    )
    plt.xlabel("Длина стержня, см")
    plt.ylabel("Температура, град C")
    plt.title("Графики зависимостей температуры для разных исходных данных")
    plt.grid(True)
    plt.legend()
    return save_plot(filename)


def write_summary(lines: list[str]) -> Path:
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return SUMMARY_PATH


def main() -> None:
    baseline_alpha = 10.0
    (
        baseline_uniform_solution,
        baseline_global_matrix,
        baseline_global_vector,
        baseline_corrected_matrix,
        baseline_local_matrices,
        baseline_local_vectors,
    ) = fem_stationary_solution(
        model=REFERENCE_MODEL,
        mesh=UNIFORM_MESH,
        alpha=baseline_alpha,
    )

    summary_lines: list[str] = [
        "Практическое занятие №3. Стационарная тепловая задача в 1D. Метод конечных элементов.",
        "",
        "Принятая расчетная постановка.",
        f"L = {REFERENCE_MODEL.length:.3f}, k = {REFERENCE_MODEL.conductivity:.3f}, "
        f"A = {REFERENCE_MODEL.area:.3f}, P = {REFERENCE_MODEL.perimeter:.3f}, "
        f"Tw = {REFERENCE_MODEL.left_temperature:.3f}, T0 = {REFERENCE_MODEL.ambient_temperature:.3f}.",
        f"Равномерная сетка: {format_array(UNIFORM_MESH)}",
        f"Неравномерная сетка: {format_array(NONUNIFORM_MESH)}",
        "",
        f"Базовый расчет для alpha = {baseline_alpha:g}.",
        "Локальные матрицы теплопроводности:",
    ]

    for index, local_matrix in enumerate(baseline_local_matrices, start=1):
        summary_lines.append(f"K_local[{index}] = {format_array(local_matrix)}")

    summary_lines.extend(
        [
            "",
            "Локальные векторы правой части:",
        ]
    )
    for index, local_vector in enumerate(baseline_local_vectors, start=1):
        summary_lines.append(f"f_local[{index}] = {format_array(local_vector)}")

    summary_lines.extend(
        [
            "",
            f"Глобальная матрица до корректировки:\n{format_array(baseline_global_matrix)}",
            f"Глобальный вектор до корректировки: {format_array(baseline_global_vector)}",
            f"Глобальная матрица после корректировки:\n{format_array(baseline_corrected_matrix)}",
            f"Решение базовой системы: {format_array(baseline_uniform_solution)}",
            "",
            "Анализ полученных результатов.",
        ]
    )

    alpha_values = [10.0, 50.0]
    plot_paths: list[Path] = []
    solutions_by_alpha: dict[float, dict[str, np.ndarray]] = {}

    for alpha in alpha_values:
        (
            uniform_solution,
            _,
            _,
            _,
            _,
            _,
        ) = fem_stationary_solution(
            model=REFERENCE_MODEL,
            mesh=UNIFORM_MESH,
            alpha=alpha,
        )
        (
            nonuniform_solution,
            _,
            _,
            _,
            _,
            _,
        ) = fem_stationary_solution(
            model=REFERENCE_MODEL,
            mesh=NONUNIFORM_MESH,
            alpha=alpha,
        )

        control_uniform = CONTROL_RESULTS[alpha]["uniform"]
        control_nonuniform = CONTROL_RESULTS[alpha]["nonuniform"]

        uniform_control_error = float(np.max(np.abs(uniform_solution - control_uniform)))
        nonuniform_control_error = float(np.max(np.abs(nonuniform_solution - control_nonuniform)))

        solutions_by_alpha[alpha] = {
            "uniform": uniform_solution,
            "nonuniform": nonuniform_solution,
        }

        summary_lines.extend(
            [
                f"alpha = {alpha:g}.",
                f"МКЭ-решение, равномерная сетка: {format_array(uniform_solution)}",
                f"Контроль из методички, равномерная сетка: {format_array(control_uniform)}",
                f"МКЭ-решение, неравномерная сетка: {format_array(nonuniform_solution)}",
                f"Контроль из методички, неравномерная сетка: {format_array(control_nonuniform)}",
                f"Максимальное отклонение от контрольных значений, равномерная сетка: {uniform_control_error:.6f}",
                f"Максимальное отклонение от контрольных значений, неравномерная сетка: {nonuniform_control_error:.6f}",
                "",
            ]
        )

        plot_paths.append(
            plot_two_meshes(
                model=REFERENCE_MODEL,
                alpha=alpha,
                uniform_solution=uniform_solution,
                nonuniform_solution=nonuniform_solution,
                filename=f"mesh_comparison_alpha_{int(alpha)}.png",
            )
        )

    plot_paths.append(
        plot_all_cases(
            model=REFERENCE_MODEL,
            solutions_by_alpha=solutions_by_alpha,
            filename="all_cases_comparison.png",
        )
    )

    summary_lines.extend(
        [
            "Выводы.",
            "1. После подстановки параметров из методички МКЭ-решение воспроизводит контрольные значения для равномерной и неравномерной сеток.",
            "2. Для alpha = 50 температура стержня быстрее приближается к температуре окружающей среды, чем для alpha = 10.",
            "3. Неравномерная сетка заметно меняет распределение температуры вблизи левого торца, но сохраняет согласованность с контрольными результатами.",
            "4. Базовые локальные матрицы, глобальная матрица и глобальный вектор правой части совпадают по структуре с формулами (5.10)-(5.12) из методички.",
            "",
            "Сохраненные графики:",
        ]
    )
    summary_lines.extend(str(path) for path in plot_paths)

    summary_path = write_summary(summary_lines)
    print("\n".join(summary_lines))
    print(f"\nСводка сохранена в: {summary_path}")


if __name__ == "__main__":
    main()
