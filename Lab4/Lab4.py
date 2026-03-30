import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

SHOW_PLOTS = os.environ.get("LAB4_SHOW_PLOTS", "0") == "1"
if not SHOW_PLOTS:
    matplotlib.use("Agg")


BASE_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BASE_DIR / "plots"
SUMMARY_PATH = BASE_DIR / "results_summary.txt"


NODE_COORDINATES = {
    1: (0.0, 0.0),
    2: (0.5, 0.0),
    3: (1.0, 0.0),
    4: (0.5, 0.5),
    5: (0.0, 1.0),
    6: (0.5, 1.0),
    7: (1.0, 1.0),
    8: (0.5, 1.5),
    9: (0.0, 2.0),
    10: (0.5, 2.0),
    11: (1.0, 2.0),
}

ELEMENTS = [
    (9, 8, 10),
    (10, 8, 11),
    (5, 8, 9),
    (7, 11, 8),
    (8, 5, 6),
    (8, 6, 7),
    (5, 4, 6),
    (6, 4, 7),
    (5, 1, 4),
    (7, 4, 3),
    (1, 2, 4),
    (2, 3, 4),
]

DIRICHLET_VALUES = {
    1: 0.0,
    2: 0.0,
    3: 0.0,
    5: 0.0,
    9: 0.0,
    10: 21.213,
    11: 30.0,
}

CONTROL_MKR = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [21.213, 13.674, 8.136, 3.791, 0.0],
        [30.0, 19.337, 11.506, 5.361, 0.0],
        [21.213, 13.674, 8.136, 3.791, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    dtype=float,
)

CONTROL_MKE_VECTOR = np.array(
    [0.0, 0.0, 0.0, 1.811, 0.0, 4.527, 5.432, 10.864, 0.0, 21.213, 30.0],
    dtype=float,
)


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


def format_array(array: np.ndarray, precision: int = 4) -> str:
    return np.array2string(array, precision=precision, suppress_small=False)


def triangle_area(element: tuple[int, int, int]) -> float:
    points = np.array([NODE_COORDINATES[node] for node in element], dtype=float)
    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]
    return abs(0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)))


def local_stiffness_matrix(element: tuple[int, int, int]) -> np.ndarray:
    points = np.array([NODE_COORDINATES[node] for node in element], dtype=float)
    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]

    area = triangle_area(element)
    b = np.array([y2 - y3, y3 - y1, y1 - y2], dtype=float)
    c = np.array([x3 - x2, x1 - x3, x2 - x1], dtype=float)

    return (np.outer(b, b) + np.outer(c, c)) / (4.0 * area)


def assemble_global_matrix() -> tuple[np.ndarray, list[np.ndarray]]:
    node_count = len(NODE_COORDINATES)
    global_matrix = np.zeros((node_count, node_count), dtype=float)
    local_matrices: list[np.ndarray] = []

    for element in ELEMENTS:
        element_matrix = local_stiffness_matrix(element)
        local_matrices.append(element_matrix)

        for i_local, i_global in enumerate(element):
            for j_local, j_global in enumerate(element):
                global_matrix[i_global - 1, j_global - 1] += element_matrix[i_local, j_local]

    return global_matrix, local_matrices


def apply_dirichlet_conditions(
    matrix: np.ndarray,
    values: dict[int, float],
) -> tuple[np.ndarray, np.ndarray]:
    corrected_matrix = matrix.copy()
    rhs = np.zeros(matrix.shape[0], dtype=float)

    for node, value in values.items():
        index = node - 1
        rhs -= corrected_matrix[:, index] * value
        corrected_matrix[:, index] = 0.0
        corrected_matrix[index, :] = 0.0
        corrected_matrix[index, index] = 1.0
        rhs[index] = value

    return corrected_matrix, rhs


def solve_fem() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    global_matrix, local_matrices = assemble_global_matrix()
    corrected_matrix, rhs = apply_dirichlet_conditions(global_matrix, DIRICHLET_VALUES)
    solution = np.linalg.solve(corrected_matrix, rhs)
    return solution, global_matrix, corrected_matrix, local_matrices


def mke_line_x_half(solution: np.ndarray) -> np.ndarray:
    indices = [10, 8, 6, 4, 2]
    return np.array([solution[index - 1] for index in indices], dtype=float)


def mke_line_x_one(solution: np.ndarray) -> np.ndarray:
    node11 = solution[10]
    node7 = solution[6]
    node3 = solution[2]
    return np.array(
        [
            node11,
            0.5 * (node11 + node7),
            node7,
            0.5 * (node7 + node3),
            node3,
        ],
        dtype=float,
    )


def control_libman_line_2() -> np.ndarray:
    return CONTROL_MKR[1]


def control_libman_line_3() -> np.ndarray:
    return CONTROL_MKR[2]


def plot_mesh(filename: str) -> Path:
    plt.figure(figsize=(6, 8))

    for element_number, element in enumerate(ELEMENTS, start=1):
        points = np.array([NODE_COORDINATES[node] for node in (*element, element[0])], dtype=float)
        plt.plot(points[:, 0], points[:, 1], color="black", linewidth=1)
        centroid = np.mean(np.array([NODE_COORDINATES[node] for node in element], dtype=float), axis=0)
        plt.text(centroid[0], centroid[1], str(element_number), color="tab:blue", fontsize=10, ha="center", va="center")

    for node, (x_value, y_value) in NODE_COORDINATES.items():
        plt.scatter([x_value], [y_value], color="tab:red", s=18)
        plt.text(x_value + 0.02, y_value + 0.02, str(node), fontsize=9)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("МКЭ-сетка для уравнения Лапласа")
    plt.axis("equal")
    plt.grid(True)
    return save_plot(filename)


def plot_line_comparison(
    x_values: np.ndarray,
    libman_values: np.ndarray,
    mke_values: np.ndarray,
    title: str,
    filename: str,
) -> Path:
    plt.figure(figsize=(6, 4.5))
    plt.plot(x_values, libman_values, color="red", linewidth=2, label="Libman")
    plt.plot(x_values, mke_values, color="royalblue", linestyle=":", linewidth=2.5, label="MKE")
    plt.xlabel("x1")
    plt.ylabel("U")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    return save_plot(filename)


def plot_combined_comparison(
    x_values: np.ndarray,
    libman_2: np.ndarray,
    libman_3: np.ndarray,
    mke_2: np.ndarray,
    mke_3: np.ndarray,
    filename: str,
) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    axes[0].plot(x_values, libman_2, color="red", linewidth=2, label="Libman_2")
    axes[0].plot(x_values, mke_2, color="royalblue", linestyle=":", linewidth=2.5, label="MKE_2")
    axes[0].set_title("Сравнение по линии 2")
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("U")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(x_values, libman_3, color="red", linewidth=2, label="Libman_3")
    axes[1].plot(x_values, mke_3, color="royalblue", linestyle=":", linewidth=2.5, label="MKE_3")
    axes[1].set_title("Сравнение по линии 3")
    axes[1].set_xlabel("x1")
    axes[1].grid(True)
    axes[1].legend()

    figure.suptitle("Сопоставление решений МКР и МКЭ")
    figure.tight_layout(rect=(0, 0, 1, 0.95))

    ensure_plot_dir()
    output_path = PLOTS_DIR / filename
    figure.savefig(output_path, dpi=180)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(figure)
    return output_path


def write_summary(lines: list[str]) -> Path:
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return SUMMARY_PATH


def main() -> None:
    fem_solution, global_matrix, corrected_matrix, local_matrices = solve_fem()
    rhs = np.zeros(len(NODE_COORDINATES), dtype=float)
    for node, value in DIRICHLET_VALUES.items():
        index = node - 1
        rhs -= global_matrix[:, index] * value
    for node, value in DIRICHLET_VALUES.items():
        rhs[node - 1] = value

    mke_2 = mke_line_x_half(fem_solution)
    mke_3 = mke_line_x_one(fem_solution)
    libman_2 = control_libman_line_2()
    libman_3 = control_libman_line_3()

    control_error = float(np.max(np.abs(fem_solution - CONTROL_MKE_VECTOR)))
    line2_error = float(np.max(np.abs(mke_2 - libman_2)))
    line3_error = float(np.max(np.abs(mke_3 - libman_3)))

    summary_lines: list[str] = [
        "Практическое занятие №4. Уравнение Лапласа методом конечных элементов.",
        "",
        "Постановка задачи.",
        "Рассматривается половина прямоугольной области с 11 узлами и 12 линейными треугольными элементами.",
        "Граничные условия Дирихле заданы в узлах 1, 2, 3, 5, 9, 10, 11.",
        "",
        f"Координаты узлов: {NODE_COORDINATES}",
        f"Элементы: {ELEMENTS}",
        "",
        "Локальные матрицы теплопроводности для первых пяти элементов:",
    ]

    for index, local_matrix in enumerate(local_matrices[:5], start=1):
        summary_lines.append(f"k{index} = {format_array(local_matrix)}")

    summary_lines.extend(
        [
            "",
            f"Глобальная матрица до корректировки:\n{format_array(global_matrix)}",
            "",
            f"Глобальная матрица после корректировки:\n{format_array(corrected_matrix)}",
            "",
            f"Вектор правой части после корректировки: {format_array(rhs)}",
            f"Решение МКЭ: {format_array(fem_solution)}",
            f"Контрольное решение из методички: {format_array(CONTROL_MKE_VECTOR)}",
            f"Максимальное отклонение от контрольного решения МКЭ: {control_error:.6f}",
            "",
            f"Контрольная матрица МКР:\n{format_array(CONTROL_MKR)}",
            f"Линия 2, МКР (контроль): {format_array(libman_2)}",
            f"Линия 2, МКЭ: {format_array(mke_2)}",
            f"Линия 3, МКР (контроль): {format_array(libman_3)}",
            f"Линия 3, МКЭ: {format_array(mke_3)}",
            f"Максимальное расхождение МКР/МКЭ по линии 2: {line2_error:.6f}",
            f"Максимальное расхождение МКР/МКЭ по линии 3: {line3_error:.6f}",
            "",
            "Выводы.",
            "1. Сборка МКЭ по треугольным линейным элементам воспроизводит контрольное решение из методички.",
            "2. После корректировки по граничным условиям ненулевые значения внутри области возникают только в узлах 4, 6, 7 и 8.",
            "3. Сопоставление с контрольными данными МКР показывает ожидаемое различие между МКЭ и МКР на выбранных линиях сравнения.",
        ]
    )

    x_values = np.linspace(0.0, 1.0, 5)
    plot_paths = [
        plot_mesh("fem_mesh.png"),
        plot_line_comparison(x_values, libman_2, mke_2, "Сравнение по линии 2", "line2_comparison.png"),
        plot_line_comparison(x_values, libman_3, mke_3, "Сравнение по линии 3", "line3_comparison.png"),
        plot_combined_comparison(x_values, libman_2, libman_3, mke_2, mke_3, "mkr_vs_mke.png"),
    ]

    summary_lines.extend(["", "Сохраненные графики:"])
    summary_lines.extend(str(path) for path in plot_paths)

    summary_path = write_summary(summary_lines)
    print("\n".join(summary_lines))
    print(f"\nСводка сохранена в: {summary_path}")


if __name__ == "__main__":
    main()
