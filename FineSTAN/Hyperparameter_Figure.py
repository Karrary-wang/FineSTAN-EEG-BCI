"""import matplotlib.pyplot as plt
import numpy as np

# -------------------------- 全局风格设置 --------------------------
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['lines.markersize'] = 8
plt.rcParams['grid.linestyle'] = '-'
plt.rcParams['grid.alpha'] = 0.6

# -------------------------- 实验数据 --------------------------
channels = ['1', '2', '4', '8', '12', '16', '20', '24', '28', 'Full']
x = np.arange(len(channels))

acc_ccd = [49.30, 60.23, 68.05, 69.55, 70.57, 71.38, 70.54, 71.25, 70.02, 69.73]
acc_sv = [45.50, 53.60, 61.04, 63.76, 65.37, 65.59, 65.65, 66.22, 66.26, 66.03]
acc_ell = [36.16, 48.91, 56.16, 59.00, 59.07, 60.70, 59.41, 58.07, 58.05, 58.13]

# -------------------------- 创建子图 --------------------------
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), dpi=300)
axes = [ax1, ax2, ax3]

plot_color = '#1f77b4'
marker_style = 'o'

# -------------------------- 绘制子图 --------------------------
for i, acc in enumerate([acc_ccd, acc_sv, acc_ell]):
    ax = axes[i]

    # 主折线
    ax.plot(x, acc, marker=marker_style, color=plot_color,
            linewidth=2.5, markersize=8, alpha=0.9, zorder=3)

    # 全通道虚线
    full_channel_acc = acc[-1]
    ax.axhline(y=full_channel_acc,
               color='#aaaaaa',
               linestyle='--',
               linewidth=1.8,
               alpha=0.6,
               zorder=2)

    # 最优值标注
    best_idx = np.argmax(acc)
    best_acc = acc[best_idx]
    offset = 1.0 if i == 2 else 0.8
    ax.text(best_idx, best_acc + offset, f'{best_acc:.2f}%',
            ha='center', va='bottom', fontweight='bold', fontsize=12, color='black')

    # 坐标轴设置
    ax.set_xticks(x)
    ax.set_xticklabels(channels, fontfamily='Arial')
    ax.grid(True, axis='both', zorder=1)

    if i == 0:  # CCS-HI
        ax.set_ylim(45, 77.5)
        ax.set_yticks(np.arange(45, 78, 5))
    elif i == 1:  # SV-HI
        ax.set_ylim(40, 75)
        ax.set_yticks(np.arange(40, 76, 5))
    else:  # ELL-HI
        ax.set_ylim(30, 65)
        ax.set_yticks(np.arange(30, 66, 5))

    # 强制Y轴刻度显示为整数
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{int(x)}'))

    if i == 0:
        ax.set_ylabel('Accuracy (%)', fontsize=16, fontfamily='Arial')

# -------------------------- 全局X轴标签 --------------------------
fig.text(0.5, 0.16, 'Number of Selected Channels',
         ha='center', va='center', fontsize=16, fontfamily='Arial')

# -------------------------- 核心修改：(a)(b)(c) 精准定位在12-16通道之间 --------------------------
for i, ax in enumerate(axes):
    # x=4.5 是数据坐标，正好对应12通道（索引4）和16通道（索引5）的正中间
    # y=-0.2 是X轴相对坐标，在X轴下方20%的位置，与全局标签间距适中
    ax.text(4.5, -0.15, f'({chr(97 + i)})',
            transform=ax.get_xaxis_transform(),  # x用数据坐标，y用X轴相对坐标
            fontsize=16, fontweight='bold', fontfamily='Arial',
            ha='center', va='top')

# -------------------------- 布局调整 --------------------------
plt.subplots_adjust(wspace=0.18)
plt.tight_layout(rect=[0, 0.05, 1, 0.96])

# -------------------------- 保存 --------------------------
plt.savefig('fig10_channel_sensitivity_final.pdf', bbox_inches='tight', pad_inches=0.05)
plt.savefig('fig10_channel_sensitivity_final.png', bbox_inches='tight', dpi=600)
plt.show()  """

import matplotlib.pyplot as plt
import numpy as np

# ---------------------- 1. 实验数据 ----------------------
window_sizes = [50, 75, 100, 125, 150, 175, 200, 225, 250]
tasks = ['CCS-HI', 'SV-HI', 'ELL-HI']

# 各窗口对应准确率
acc_data = {
    'CCS-HI': [71.38, 70.23, 70.61, 69.56, 70.73, 69.41, 70.76, 69.52, 69.97],
    'SV-HI':  [65.25, 65.41, 66.26, 65.56, 65.37, 65.38, 65.13, 65.07, 65.07],
    'ELL-HI': [60.45, 59.39, 60.54, 59.73, 59.93, 60.43, 60.70, 60.32, 60.23]
}

# 无SegSTA模块基线准确率（正确对照值）
baseline_acc = {
    'CCS-HI': 69.47,
    'SV-HI':  65.17,
    'ELL-HI': 59.44
}

# 各任务最优窗口索引
optimal_indices = {
    'CCS-HI': 0,
    'SV-HI':  2,
    'ELL-HI': 6
}

# ---------------------- 2. 全局样式配置（整体放大字体） ----------------------
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12        # 全局基础字体放大
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# 配色
bar_color = '#1f77b4'
highlight_color = '#ff4444'
baseline_color = '#888888'

# ---------------------- 3. 绘图主体 ----------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=120)
fig.subplots_adjust(wspace=0.3, bottom=0.18)

for i, task in enumerate(tasks):
    ax = axes[i]
    acc = acc_data[task]
    x = np.arange(len(window_sizes))
    bar_width = 0.6

    # 绘制柱状图
    bars = ax.bar(
        x, acc,
        width=bar_width,
        color=bar_color,
        edgecolor='black',
        linewidth=0.5,
        alpha=0.8
    )

    # 最优柱红色粗边框高亮
    opt_idx = optimal_indices[task]
    bars[opt_idx].set_edgecolor(highlight_color)
    bars[opt_idx].set_linewidth(2)

    # ========== 核心改动：仅最优窗口标注数值，其余删除 ==========
    for idx, bar in enumerate(bars):
        if idx == opt_idx:
            height = bar.get_height()
            text_offset = 0.25 if task == 'ELL-HI' else 0.2
            ax.text(
                bar.get_x() + bar.get_width()/2., height + text_offset,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=11  # 数值字体放大
            )

    # 绘制基线虚线
    bl = baseline_acc[task]
    ax.axhline(
        y=bl,
        color=baseline_color,
        linestyle='--',
        alpha=0.7
    )

    # 坐标轴 & 刻度（字体统一放大）
    ax.set_xticks(x)
    ax.set_xticklabels(window_sizes, fontsize=11)
    ax.set_xlabel('Window Size', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)

    # 分任务纵轴范围+刻度（保持原有最优设置）
    if task == 'CCS-HI':
        ax.set_ylim(50, 75)
        ax.set_yticks([50, 55, 60, 65, 70, 75])
        ax.tick_params(axis='y', labelsize=11)
    elif task == 'SV-HI':
        ax.set_ylim(50, 70)
        ax.set_yticks([50, 54, 58, 62, 66, 70])
        ax.tick_params(axis='y', labelsize=11)
    elif task == 'ELL-HI':
        ax.set_ylim(50, 62)
        ax.set_yticks([50, 52, 54, 56, 58, 60, 62])
        ax.tick_params(axis='y', labelsize=11)

    # ========== 核心改动：(a)(b)(c) 放大 + 加粗 ==========
    ax.text(
        0.5, -0.14, f'({chr(97+i)})',
        ha='center', va='top',
        transform=ax.transAxes,
        fontsize=14,    # 大幅放大
        fontweight='bold' # 加粗
    )

plt.tight_layout()
plt.savefig('segsta_window_bar_final.png', dpi=300, bbox_inches='tight')
plt.show()