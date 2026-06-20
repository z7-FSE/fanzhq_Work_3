"""《人工智能编程语言》第三次作业：公交 IC 卡刷卡数据分析。

姓名：范哲齐
学号：25361077
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "ICData.csv"


def print_task_title(number, title):
    """打印统一的任务分隔标题。"""
    print("\n" + "=" * 72)
    print(f"任务{number}  {title}")
    print("=" * 72)


def load_and_preprocess(csv_path=DATA_FILE):
    """读取 IC 卡数据，完成时间解析、衍生字段构造和异常值清理。"""
    print_task_title(1, "数据预处理")

    # 使用 utf-8-sig 自动处理 CSV 表头可能带有的 BOM 字符。
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    print("数据集前5行：")
    print(df.head().to_string(index=False))
    print(f"\n基本信息：{df.shape[0]} 行，{df.shape[1]} 列")
    print("各列初始数据类型：")
    print(df.dtypes.to_string())

    # errors='coerce' 会把无法解析的时间置为 NaT，便于后续统一当作缺失值处理。
    df["交易时间"] = pd.to_datetime(df["交易时间"], errors="coerce")
    # dt.hour 从每个 datetime 中提取 0~23 的小时整数。
    df["hour"] = df["交易时间"].dt.hour

    # 下车站点与上车站点之差取绝对值，得到搭乘站点数。
    df["ride_stops"] = (df["下车站点"] - df["上车站点"]).abs()
    zero_stop_mask = df["ride_stops"].eq(0)
    deleted_zero_stops = int(zero_stop_mask.sum())
    df = df.loc[~zero_stop_mask].copy()
    print(f"\n删除 ride_stops=0 的异常记录：{deleted_zero_stops} 行")

    print("各列缺失值数量：")
    print(df.isna().sum().to_string())
    before_dropna = len(df)
    df = df.dropna().copy()
    deleted_missing = before_dropna - len(df)
    print(f"删除含缺失值的记录：{deleted_missing} 行")

    # 缺失值已删除，将小时列转回普通整数，便于 NumPy 统计和绘图。
    df["hour"] = df["hour"].astype(int)
    print(f"预处理后有效记录：{len(df)} 行")
    return df


def analyze_time_distribution(df):
    """使用 NumPy 统计特定时段，并绘制 24 小时刷卡量柱状图。"""
    print_task_title(2, "时间分布分析")

    boardings = df.loc[df["刷卡类型"].eq(0)].copy()
    hours = boardings["hour"].to_numpy(dtype=int)

    # 通过 NumPy 布尔索引选出早峰前与深夜的小时值，再用元素数量得到刷卡量。
    early_hours = hours[hours < 7]
    late_hours = hours[hours >= 22]
    early_count = int(early_hours.size)
    late_count = int(late_hours.size)
    total_count = int(hours.size)

    early_pct = early_count / total_count * 100 if total_count else 0.0
    late_pct = late_count / total_count * 100 if total_count else 0.0
    print(f"全天上车刷卡量：{total_count} 次")
    print(f"早峰前时段（hour < 7）：{early_count} 次，占比 {early_pct:.2f}%")
    print(f"深夜时段（hour >= 22）：{late_count} 次，占比 {late_pct:.2f}%")

    # bincount 直接对 0~23 的小时整数计数，minlength 确保无数据的小时也保留。
    hourly_counts = np.bincount(hours, minlength=24)[:24]
    hour_values = np.arange(24)
    bar_colors = np.where(
        hour_values < 7,
        "#4C78A8",
        np.where(hour_values >= 22, "#E45756", "#72B7B2"),
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(hour_values, hourly_counts, color=bar_colors, edgecolor="white", linewidth=0.7)
    ax.set_title("24-Hour Boarding Distribution", fontsize=15, pad=12)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Number of Boardings")
    ax.set_xticks(np.arange(0, 24, 2))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(color="#4C78A8", label="Before 07:00"),
            Patch(color="#72B7B2", label="07:00-21:59"),
            Patch(color="#E45756", label="22:00 and later"),
        ],
        frameon=False,
    )
    fig.tight_layout()
    output_path = PROJECT_DIR / "hour_distribution.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"24小时分布图已保存：{output_path}")
    plt.show()
    plt.close(fig)
    return boardings, hourly_counts


def analyze_route_stops(df, route_col='线路号', stops_col='ride_stops'):
    """
    计算各线路乘客的平均搭乘站点数及其标准差。
    Parameters
    ----------
    df : pd.DataFrame  预处理后的数据集
    route_col : str    线路号列名
    stops_col : str    搭乘站点数列名
    Returns
    -------
    pd.DataFrame  包含列：线路号、mean_stops、std_stops，按 mean_stops 降序排列
    """
    route_stats = (
        df.groupby(route_col, as_index=False)[stops_col]
        .agg(mean_stops="mean", std_stops="std")
        .sort_values("mean_stops", ascending=False)
        .reset_index(drop=True)
    )
    # 若某条线路只有1条记录，样本标准差为 NaN，此时按 0 处理便于绘制误差棒。
    route_stats["std_stops"] = route_stats["std_stops"].fillna(0.0)
    return route_stats


def plot_route_stops(boardings):
    """打印线路统计结果，并使用 seaborn 绘制均值最高的前15条线路。"""
    print_task_title(3, "线路站点分析")
    route_stats = analyze_route_stops(boardings)
    print("各线路搭乘站点数统计（前10行）：")
    print(route_stats.head(10).to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    top15 = route_stats.head(15).copy()
    top15["线路标签"] = top15["线路号"].astype(str)
    top_route_labels = top15["线路标签"].tolist()
    plot_records = boardings.loc[boardings["线路号"].isin(top15["线路号"])].copy()
    plot_records["线路标签"] = plot_records["线路号"].astype(str)

    fig, ax = plt.subplots(figsize=(10, 7))
    barplot_options = dict(
        data=plot_records,
        x="ride_stops",
        y="线路标签",
        order=top_route_labels,
        estimator=np.mean,
        capsize=0.3,
        palette="Blues_d",
        ax=ax,
    )
    try:
        # seaborn 0.12+ 使用 errorbar='sd' 直接展示各线路的标准差。
        sns.barplot(**barplot_options, errorbar="sd")
    except (TypeError, AttributeError):
        # 兼容 seaborn 0.11 及更早版本的 ci 参数写法。
        sns.barplot(**barplot_options, ci="sd")

    ax.set_title("Top 15 Routes by Average Ride Stops", fontsize=15, pad=12)
    ax.set_xlabel("Average Number of Ride Stops")
    ax.set_ylabel("Route")
    ax.set_xlim(left=0)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    output_path = PROJECT_DIR / "route_stops.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"线路站点分析图已保存：{output_path}")
    plt.show()
    plt.close(fig)
    return route_stats


def main():
    """按任务顺序执行整个分析流程。"""
    sns.set_theme(style="whitegrid", context="notebook")
    df = load_and_preprocess()
    boardings, _ = analyze_time_distribution(df)
    plot_route_stops(boardings)


if __name__ == "__main__":
    main()
