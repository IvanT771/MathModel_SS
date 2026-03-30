import math
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

SHOW_PLOTS = os.environ.get("LAB2_SHOW_PLOTS", "0") == "1"
if not SHOW_PLOTS:
    matplotlib.use("Agg")


BASE_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BASE_DIR / "plots"
SUMMARY_PATH = BASE_DIR / "results_summary.txt"


@dataclass(frozen=True)
class HeatRodModel:
    length: float
    space_intervals: int
    thermal_conductivity: float
    density: float
    heat_capacity: float

    @property
    def diffusivity(self) -> float:
        return self.thermal_conductivity / (self.density * self.heat_capacity)

    @property
    def dx(self) -> float:
        return self.length / self.space_intervals

    @property
    def node_count(self) -> int:
        return self.space_intervals + 1


# Набор параметров согласован с контрольными стационарными значениями из методички.
REFERENCE_MODEL = HeatRodModel(
    length=1.0,
    space_intervals=5,
    thermal_conductivity=2.0 / 15.0,
    density=0.1,
    heat_capacity=1.0,
)


def build_x_grid(model: HeatRodModel) -> np.ndarray:
    return np.linspace(0.0, model.length, model.node_count)


def stability_factor(model: HeatRodModel, dt: float) -> float:
    return model.diffusivity * dt / (model.dx * model.dx)


def explicit_dirichlet(
    model: HeatRodModel,
    left_temperature: float,
    right_temperature: float,
    initial_temperature: float,
    t_max: float,
    time_intervals: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    x = build_x_grid(model)
    dt = t_max / time_intervals
    r = stability_factor(model, dt)

    history = np.zeros((time_intervals + 1, model.node_count), dtype=float)
    history[0, :] = initial_temperature
    history[0, 0] = left_temperature
    history[0, -1] = right_temperature

    for step in range(time_intervals):
        previous = history[step]
        current = history[step + 1]

        current[0] = left_temperature
        current[-1] = right_temperature
        current[1:-1] = (
            previous[1:-1]
            + r * (previous[2:] - 2.0 * previous[1:-1] + previous[:-2])
        )

    times = np.linspace(0.0, t_max, time_intervals + 1)
    return x, times, history, r


def explicit_robin(
    model: HeatRodModel,
    left_environment_temperature: float,
    left_alpha: float,
    right_environment_temperature: float,
    right_alpha: float,
    initial_temperature: float,
    t_max: float,
    time_intervals: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    x = build_x_grid(model)
    dt = t_max / time_intervals
    r = stability_factor(model, dt)

    history = np.zeros((time_intervals + 1, model.node_count), dtype=float)
    history[0, :] = initial_temperature

    left_denominator = (model.thermal_conductivity / model.dx) + left_alpha
    right_denominator = (model.thermal_conductivity / model.dx) + right_alpha

    for step in range(time_intervals):
        previous = history[step]
        current = history[step + 1]

        current[1:-1] = (
            previous[1:-1]
            + r * (previous[2:] - 2.0 * previous[1:-1] + previous[:-2])
        )

        current[0] = (
            (model.thermal_conductivity / model.dx) * current[1]
            + left_alpha * left_environment_temperature
        ) / left_denominator
        current[-1] = (
            (model.thermal_conductivity / model.dx) * current[-2]
            + right_alpha * right_environment_temperature
        ) / right_denominator

    times = np.linspace(0.0, t_max, time_intervals + 1)
    return x, times, history, r


def stationary_dirichlet_solution(
    x: np.ndarray,
    left_temperature: float,
    right_temperature: float,
) -> np.ndarray:
    return left_temperature + (right_temperature - left_temperature) * x / x[-1]


def analytical_transient_dirichlet_solution(
    x: np.ndarray,
    time_value: float,
    diffusivity: float,
    left_temperature: float,
    right_temperature: float,
    initial_temperature: float,
    n_terms: int = 400,
) -> np.ndarray:
    length = x[-1]
    steady = stationary_dirichlet_solution(x, left_temperature, right_temperature)

    a0 = initial_temperature - left_temperature
    b0 = (left_temperature - right_temperature) / length

    result = steady.copy()

    for n in range(1, n_terms + 1):
        coefficient = (
            2.0
            * (a0 * (1.0 - (-1) ** n) - b0 * length * (-1) ** n)
            / (n * math.pi)
        )
        result += (
            coefficient
            * np.sin(n * math.pi * x / length)
            * math.exp(-diffusivity * (n * math.pi / length) ** 2 * time_value)
        )

    return result


def stationary_robin_solution(
    x: np.ndarray,
    model: HeatRodModel,
    left_environment_temperature: float,
    left_alpha: float,
    right_environment_temperature: float,
    right_alpha: float,
) -> tuple[np.ndarray, float]:
    heat_flux = (
        left_environment_temperature - right_environment_temperature
    ) / (
        (1.0 / left_alpha)
        + (model.length / model.thermal_conductivity)
        + (1.0 / right_alpha)
    )

    left_boundary = left_environment_temperature - heat_flux / left_alpha
    profile = left_boundary - (heat_flux / model.thermal_conductivity) * x
    return profile, heat_flux


def nearest_time_indices(times: np.ndarray, target_times: list[float]) -> dict[float, int]:
    mapping: dict[float, int] = {}
    for target_time in target_times:
        index = int(np.argmin(np.abs(times - target_time)))
        mapping[target_time] = index
    return mapping


def max_abs_error(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.max(np.abs(first - second)))


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


def plot_profiles(
    x: np.ndarray,
    profiles: list[tuple[str, np.ndarray]],
    title: str,
    filename: str,
    analytical_profile: np.ndarray | None = None,
    analytical_label: str | None = None,
) -> Path:
    plt.figure(figsize=(8, 5))
    for label, profile in profiles:
        plt.plot(x, profile, marker="o", linewidth=2, label=label)

    if analytical_profile is not None and analytical_label is not None:
        plt.plot(
            x,
            analytical_profile,
            linestyle="--",
            color="black",
            linewidth=2,
            label=analytical_label,
        )

    plt.xlabel("x")
    plt.ylabel("U(x, t)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    return save_plot(filename)


def plot_time_step_comparison(
    x: np.ndarray,
    profiles_300: dict[float, np.ndarray],
    profiles_600: dict[float, np.ndarray],
    filename: str,
) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    axes = axes.ravel()

    for axis, target_time in zip(axes, profiles_300):
        axis.plot(x, profiles_300[target_time], marker="o", linewidth=2, label="m=300")
        axis.plot(x, profiles_600[target_time], marker="s", linewidth=2, label="m=600")
        axis.set_title(f"t ≈ {target_time:.3f} c")
        axis.grid(True)
        axis.legend()

    for axis in axes[2:]:
        axis.set_xlabel("x")
    axes[0].set_ylabel("U(x, t)")
    axes[2].set_ylabel("U(x, t)")

    figure.suptitle("Сопоставление решений для m=300 и m=600", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.96))

    ensure_plot_dir()
    output_path = PLOTS_DIR / filename
    figure.savefig(output_path, dpi=180)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(figure)
    return output_path


def verification_task(model: HeatRodModel) -> tuple[list[str], list[Path]]:
    target_times = [4.0, 10.0, 20.0, 50.0]
    x, times, history, r = explicit_dirichlet(
        model=model,
        left_temperature=100.0,
        right_temperature=40.0,
        initial_temperature=40.0,
        t_max=50.0,
        time_intervals=7500,
    )

    indices = nearest_time_indices(times, target_times)
    profiles: list[tuple[str, np.ndarray]] = []
    summary_lines = [
        "Задание 1. Верификация модели (ГУ 1-го рода, аналитическое решение).",
        f"Параметры модели: dx = {model.dx:.5f}, a = {model.diffusivity:.5f}.",
        f"Показатель устойчивости r = {r:.5f}.",
    ]

    for target_time in target_times:
        index = indices[target_time]
        analytical = analytical_transient_dirichlet_solution(
            x=x,
            time_value=times[index],
            diffusivity=model.diffusivity,
            left_temperature=100.0,
            right_temperature=40.0,
            initial_temperature=40.0,
        )
        error = max_abs_error(history[index], analytical)
        profiles.append((f"t = {times[index]:.0f} c", history[index]))
        summary_lines.append(
            f"Максимальная ошибка при t = {times[index]:.0f} c: {error:.6e}."
        )

    final_stationary = stationary_dirichlet_solution(x, 100.0, 40.0)
    stationary_error = max_abs_error(history[-1], final_stationary)
    summary_lines.append(
        f"Отклонение от стационарного линейного решения при t = 50 c: {stationary_error:.6e}."
    )

    plot_path = plot_profiles(
        x=x,
        profiles=profiles,
        title="Верификация явной схемы при ГУ 1-го рода",
        filename="verification_dirichlet.png",
        analytical_profile=final_stationary,
        analytical_label="Стационарное аналитическое решение",
    )

    return summary_lines, [plot_path]


def stability_task(model: HeatRodModel) -> tuple[list[str], list[Path]]:
    x, stable_times, stable_history, stable_r = explicit_dirichlet(
        model=model,
        left_temperature=100.0,
        right_temperature=40.0,
        initial_temperature=40.0,
        t_max=2.0,
        time_intervals=300,
    )

    unstable_model = HeatRodModel(
        length=model.length,
        space_intervals=model.space_intervals,
        thermal_conductivity=model.thermal_conductivity * 10.0,
        density=model.density,
        heat_capacity=model.heat_capacity,
    )
    _, unstable_times, unstable_history, unstable_r = explicit_dirichlet(
        model=unstable_model,
        left_temperature=100.0,
        right_temperature=40.0,
        initial_temperature=40.0,
        t_max=2.0,
        time_intervals=300,
    )

    stable_indices = nearest_time_indices(stable_times, [0.333, 0.667, 1.333, 2.0])
    unstable_indices = nearest_time_indices(unstable_times, [0.007, 0.013, 0.020, 0.033])

    profiles: list[tuple[str, np.ndarray]] = []
    for target_time, index in stable_indices.items():
        profiles.append((f"Устойчивая схема, t ≈ {target_time:.3f}", stable_history[index]))
    for target_time, index in unstable_indices.items():
        profiles.append((f"Неустойчивая схема, t ≈ {target_time:.3f}", unstable_history[index]))

    unstable_norm = float(np.max(np.abs(unstable_history[unstable_indices[0.033]])))

    summary_lines = [
        "Задание 2. Проверка устойчивости явной схемы.",
        f"Базовый случай: r = {stable_r:.5f} <= 0.5, схема устойчива.",
        f"При увеличении λ в 10 раз: r = {unstable_r:.5f} > 0.5, схема неустойчива.",
        f"Максимальная по модулю температура в неустойчивом режиме к t ≈ 0.033 c: {unstable_norm:.3e}.",
    ]

    plot_path = plot_profiles(
        x=x,
        profiles=profiles,
        title="Проверка устойчивости явной схемы",
        filename="stability_check.png",
    )

    return summary_lines, [plot_path]


def first_kind_time_step_task(model: HeatRodModel) -> tuple[list[str], list[Path]]:
    target_times = [0.027, 0.067, 0.133, 0.333]
    x, times_300, history_300, r_300 = explicit_dirichlet(
        model=model,
        left_temperature=150.0,
        right_temperature=40.0,
        initial_temperature=40.0,
        t_max=2.0,
        time_intervals=300,
    )
    _, times_600, history_600, r_600 = explicit_dirichlet(
        model=model,
        left_temperature=150.0,
        right_temperature=40.0,
        initial_temperature=40.0,
        t_max=2.0,
        time_intervals=600,
    )

    indices_300 = nearest_time_indices(times_300, target_times)
    indices_600 = nearest_time_indices(times_600, target_times)

    profiles_300 = {
        target_time: history_300[index]
        for target_time, index in indices_300.items()
    }
    profiles_600 = {
        target_time: history_600[index]
        for target_time, index in indices_600.items()
    }

    path_300 = plot_profiles(
        x=x,
        profiles=[
            (f"t ≈ {target_time:.3f} c", profile)
            for target_time, profile in profiles_300.items()
        ],
        title="ГУ 1-го рода: решение для m = 300",
        filename="first_kind_m300.png",
    )
    path_600 = plot_profiles(
        x=x,
        profiles=[
            (f"t ≈ {target_time:.3f} c", profile)
            for target_time, profile in profiles_600.items()
        ],
        title="ГУ 1-го рода: решение для m = 600",
        filename="first_kind_m600.png",
    )
    comparison_path = plot_time_step_comparison(
        x=x,
        profiles_300=profiles_300,
        profiles_600=profiles_600,
        filename="first_kind_m300_vs_m600.png",
    )

    summary_lines = [
        "Задание 3. Исследование влияния временного шага при ГУ 1-го рода.",
        f"Для m = 300: dt = {2.0 / 300:.5f}, r = {r_300:.5f}.",
        f"Для m = 600: dt = {2.0 / 600:.5f}, r = {r_600:.5f}.",
    ]

    for target_time in target_times:
        difference = max_abs_error(profiles_300[target_time], profiles_600[target_time])
        summary_lines.append(
            f"Максимальное расхождение между m=300 и m=600 при t ≈ {target_time:.3f} c: {difference:.6f}."
        )

    return summary_lines, [path_300, path_600, comparison_path]


def third_kind_task(model: HeatRodModel) -> tuple[list[str], list[Path]]:
    target_times = [0.027, 0.067, 0.133, 0.200, 0.667, 1.333, 2.000]
    cases = [
        ("third_kind_case_1.png", "T1=150, α1=1; T2=40, α2=1", 150.0, 1.0, 40.0, 1.0),
        ("third_kind_case_2.png", "T1=150, α1=0.1; T2=40, α2=1", 150.0, 0.1, 40.0, 1.0),
        # В методичке подпись к рис. 2.13 противоречит приведенным стационарным значениям.
        # Параметры ниже воспроизводят именно численные значения из текста.
        ("third_kind_case_3.png", "T1=150, α1=1; T2=40, α2=0.1", 150.0, 1.0, 40.0, 0.1),
        ("third_kind_case_4.png", "T1=150, α1=0.1; T2=40, α2=0.1", 150.0, 0.1, 40.0, 0.1),
    ]

    summary_lines = [
        "Задание 4. Исследование процесса при ГУ 3-го рода.",
    ]
    plot_paths: list[Path] = []

    for filename, label, left_t, left_alpha, right_t, right_alpha in cases:
        x, times, history, r = explicit_robin(
            model=model,
            left_environment_temperature=left_t,
            left_alpha=left_alpha,
            right_environment_temperature=right_t,
            right_alpha=right_alpha,
            initial_temperature=40.0,
            t_max=2.0,
            time_intervals=300,
        )

        indices = nearest_time_indices(times, target_times)
        profiles = [
            (f"t ≈ {target_time:.3f} c", history[index])
            for target_time, index in indices.items()
        ]

        stationary_profile, heat_flux = stationary_robin_solution(
            x=x,
            model=model,
            left_environment_temperature=left_t,
            left_alpha=left_alpha,
            right_environment_temperature=right_t,
            right_alpha=right_alpha,
        )

        numerical_left = float(history[-1, 0])
        numerical_right = float(history[-1, -1])
        analytical_left = float(stationary_profile[0])
        analytical_right = float(stationary_profile[-1])

        summary_lines.append(
            f"{label}: r = {r:.5f}, q = {heat_flux:.5f}, "
            f"u_left(2 c) = {numerical_left:.3f}, u_right(2 c) = {numerical_right:.3f}, "
            f"аналитика = ({analytical_left:.3f}; {analytical_right:.3f})."
        )

        plot_paths.append(
            plot_profiles(
                x=x,
                profiles=profiles,
                title=f"ГУ 3-го рода: {label}",
                filename=filename,
                analytical_profile=stationary_profile,
                analytical_label="Стационарное аналитическое решение",
            )
        )

    summary_lines.append(
        "Стационарные значения для случаев α1=α2=1, α1=0.1/α2=1, α1=1/α2=0.1 и α1=α2=0.1 "
        "согласуются с формулами тепловых сопротивлений."
    )
    summary_lines.append(
        "Для случая α1=α2=0.1 к t = 2 c решение еще не полностью достигает стационарного режима, "
        "что видно по малому, но ненулевому отличию от аналитического предела."
    )

    return summary_lines, plot_paths


def write_summary(summary_lines: list[str]) -> Path:
    SUMMARY_PATH.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return SUMMARY_PATH


def main() -> None:
    all_summary_lines: list[str] = [
        "Практическое занятие №2. Уравнение параболического типа. Явная схема.",
        "",
        f"Параметры базовой модели: L = {REFERENCE_MODEL.length}, "
        f"n = {REFERENCE_MODEL.space_intervals}, dx = {REFERENCE_MODEL.dx:.5f}, "
        f"λ = {REFERENCE_MODEL.thermal_conductivity:.5f}, "
        f"ρ = {REFERENCE_MODEL.density:.5f}, c = {REFERENCE_MODEL.heat_capacity:.5f}, "
        f"a = {REFERENCE_MODEL.diffusivity:.5f}.",
        "",
    ]
    all_plot_paths: list[Path] = []

    for task_function in (
        verification_task,
        stability_task,
        first_kind_time_step_task,
        third_kind_task,
    ):
        summary_lines, plot_paths = task_function(REFERENCE_MODEL)
        all_summary_lines.extend(summary_lines)
        all_summary_lines.append("")
        all_plot_paths.extend(plot_paths)

    all_summary_lines.extend(
        [
            "Выводы.",
            "1. Явная схема корректно воспроизводит аналитическое решение при выполнении критерия устойчивости.",
            "2. Условие r <= 0.5 является определяющим: при увеличении коэффициента теплопроводности в 10 раз схема расходится.",
            "3. Уменьшение временного шага с m=300 до m=600 меняет решение слабо, то есть модель демонстрирует сходимость.",
            "4. При ГУ 3-го рода стационарные температуры на границах зависят от дополнительных тепловых сопротивлений 1/α.",
            "5. Чем меньше коэффициент теплоотдачи α, тем сильнее температура на соответствующей границе отходит от температуры внешней среды.",
            "",
            "Сохраненные графики:",
        ]
    )
    all_summary_lines.extend(str(path) for path in all_plot_paths)

    summary_path = write_summary(all_summary_lines)
    print("\n".join(all_summary_lines))
    print(f"\nСводка сохранена в: {summary_path}")


if __name__ == "__main__":
    main()
