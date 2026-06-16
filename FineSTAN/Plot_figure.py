"""import numpy as np
import matplotlib.pyplot as plt

# ====================== 1. 数据定义 + 平均值计算 ======================
subject_names = [f"S{i:02d}" for i in range(1, 22)] + ["Average"]
x_axis = np.arange(len(subject_names))  # 21个被试 + 1个平均值

# ---------------------- CCS-HI 数据 ----------------------
data_ccs = np.array([
    [64.50, 60.00, 62.00, 66.00, 65.50, 60.50, 60.50, 54.00, 67.50, 63.50, 62.50, 77.00],
    [52.50, 47.00, 66.00, 53.00, 51.00, 51.50, 61.50, 55.00, 66.00, 68.00, 59.00, 71.00],
    [44.50, 30.50, 39.00, 35.50, 38.50, 36.50, 38.00, 28.00, 38.50, 38.50, 41.00, 44.00],
    [35.50, 31.50, 40.00, 36.50, 36.50, 45.50, 41.50, 30.50, 44.00, 43.00, 41.00, 51.00],
    [36.00, 33.00, 40.00, 31.50, 34.00, 34.00, 39.00, 36.50, 37.50, 38.50, 37.00, 37.00],
    [64.00, 60.00, 74.50, 73.50, 73.50, 65.00, 73.00, 55.00, 75.00, 69.50, 71.50, 86.00],
    [33.50, 32.50, 43.50, 39.50, 40.00, 38.00, 36.00, 29.50, 37.00, 37.50, 40.50, 45.50],
    [87.00, 71.00, 79.50, 74.50, 78.00, 75.00, 83.00, 46.00, 78.00, 82.00, 75.00, 83.50],
    [74.50, 64.50, 74.50, 62.50, 64.50, 61.00, 76.00, 62.50, 79.50, 76.50, 73.00, 87.50],
    [54.32, 54.32, 64.20, 39.50, 57.41, 62.35, 67.90, 48.77, 72.22, 70.99, 63.58, 70.99],
    [64.50, 47.00, 59.00, 42.50, 79.00, 79.00, 81.50, 72.00, 86.50, 82.50, 84.50, 92.00],
    [39.50, 41.00, 50.50, 45.50, 61.00, 55.00, 58.50, 57.50, 65.50, 65.00, 54.50, 71.00],
    [88.50, 72.50, 83.50, 76.50, 54.50, 45.50, 52.00, 45.50, 56.50, 53.00, 53.00, 63.50],
    [70.50, 55.50, 71.50, 65.00, 74.50, 75.50, 80.50, 69.00, 76.50, 77.50, 74.00, 76.00],
    [77.50, 60.50, 73.50, 81.50, 78.50, 71.50, 75.00, 57.50, 77.50, 78.00, 71.50, 81.50],
    [84.50, 74.50, 82.00, 82.50, 85.50, 82.00, 85.50, 73.50, 88.00, 85.50, 79.50, 90.50],
    [56.00, 35.00, 46.50, 48.50, 43.50, 42.00, 48.50, 37.00, 49.00, 50.50, 43.00, 58.50],
    [76.50, 58.50, 81.00, 78.50, 74.50, 69.50, 71.00, 63.50, 76.50, 72.00, 73.00, 84.50],
    [62.50, 62.50, 74.00, 48.00, 69.00, 64.00, 68.00, 56.50, 74.50, 75.50, 70.50, 77.00],
    [54.50, 53.00, 56.00, 50.50, 62.00, 47.50, 49.00, 43.50, 54.50, 51.50, 53.00, 77.00],
    [73.50, 51.50, 63.50, 70.50, 63.50, 56.50, 66.50, 52.50, 62.50, 68.00, 70.50, 74.00]
])
baseline_ccs = data_ccs[:, :11]
ours_ccs = data_ccs[:, -1]
baseline_max_ccs = np.max(baseline_ccs, axis=1)
baseline_min_ccs = np.min(baseline_ccs, axis=1)
# 拼接平均值
baseline_max_ccs = np.append(baseline_max_ccs, np.mean(baseline_max_ccs))
baseline_min_ccs = np.append(baseline_min_ccs, np.mean(baseline_min_ccs))
ours_ccs = np.append(ours_ccs, np.mean(ours_ccs))

# ---------------------- SV-HI 数据 ----------------------
data_sv = np.array([
    [34.58, 27.08, 38.33, 44.58, 32.08, 29.17, 35.00, 28.33, 32.92, 35.42, 30.00, 42.08],
    [51.67, 50.42, 60.00, 50.83, 49.17, 51.50, 59.58, 51.25, 63.75, 62.08, 50.83, 67.50],
    [34.17, 27.50, 41.25, 29.17, 32.29, 40.42, 38.75, 32.50, 41.25, 40.42, 35.42, 45.42],
    [29.33, 29.78, 38.67, 30.22, 38.22, 31.11, 43.11, 32.00, 34.67, 38.22, 35.56, 44.44],
    [29.58, 26.67, 31.25, 20.00, 27.08, 29.58, 31.25, 23.33, 32.92, 30.83, 31.25, 38.75],
    [65.42, 65.83, 74.57, 67.50, 71.67, 70.00, 79.17, 64.17, 78.33, 77.50, 70.83, 83.33],
    [42.00, 44.58, 60.83, 49.17, 58.75, 52.08, 52.33, 37.50, 53.75, 53.75, 52.92, 65.00],
    [70.42, 58.75, 66.25, 65.42, 67.92, 75.00, 82.50, 68.75, 81.25, 85.00, 84.17, 78.33],
    [69.58, 68.33, 82.92, 57.50, 70.00, 75.00, 80.42, 64.58, 79.58, 72.92, 71.25, 83.33],
    [60.00, 50.42, 60.42, 30.83, 55.00, 57.08, 65.42, 60.00, 65.42, 65.00, 59.17, 71.67],
    [67.92, 61.25, 74.17, 55.83, 64.17, 61.25, 64.58, 57.50, 67.50, 64.58, 65.42, 74.17],
    [60.83, 56.67, 57.92, 49.58, 63.33, 62.50, 64.17, 55.83, 63.33, 66.67, 55.00, 70.83],
    [74.58, 81.25, 80.00, 55.00, 77.50, 75.42, 82.92, 67.08, 82.08, 79.17, 75.42, 87.50],
    [45.00, 42.92, 55.42, 40.83, 50.00, 47.92, 52.92, 38.33, 55.42, 53.75, 52.92, 57.92],
    [54.50, 42.50, 57.50, 57.92, 53.33, 50.83, 54.58, 45.42, 56.25, 55.83, 50.83, 58.75],
    [71.25, 72.50, 77.08, 70.83, 75.83, 77.08, 76.67, 72.08, 78.33, 74.17, 73.75, 81.67],
    [45.42, 46.67, 55.00, 33.33, 45.83, 47.92, 51.25, 40.83, 52.92, 51.67, 50.42, 61.25],
    [63.75, 55.42, 67.08, 68.33, 65.00, 60.42, 64.17, 53.33, 70.00, 70.00, 62.50, 74.17],
    [57.92, 62.92, 77.50, 49.17, 71.67, 71.67, 71.67, 70.00, 79.58, 74.17, 65.00, 79.58],
    [45.83, 37.08, 53.75, 62.50, 48.33, 46.67, 53.75, 38.33, 48.75, 47.08, 41.67, 67.92],
    [38.75, 43.33, 52.92, 39.17, 35.83, 42.92, 44.58, 37.08, 50.00, 44.17, 43.75, 57.92]
])
baseline_sv = data_sv[:, :11]
ours_sv = data_sv[:, -1]
baseline_max_sv = np.max(baseline_sv, axis=1)
baseline_min_sv = np.min(baseline_sv, axis=1)
# 拼接平均值
baseline_max_sv = np.append(baseline_max_sv, np.mean(baseline_max_sv))
baseline_min_sv = np.append(baseline_min_sv, np.mean(baseline_min_sv))
ours_sv = np.append(ours_sv, np.mean(ours_sv))

# ====================== 2. 最终字体配置（再大1个字号） ======================
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 16,  # 全局基础字体从15→16（+1）
    'axes.linewidth': 1.4,
    'xtick.major.width': 1.4,
    'ytick.major.width': 1.4,
    'figure.dpi': 100,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
})

# 图尺寸保持16×10不变
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

# ====================== 3. 绘制(a) CCS-HI ======================
ax1.fill_between(x_axis, baseline_min_ccs, baseline_max_ccs, color='#FFE0B2', alpha=0.5, label='ACC Range (others)')
ax1.plot(x_axis, ours_ccs, color='#FF8800', marker='o', markersize=8, lw=2.1, label='FineSTAN ACC')

# 所有字体统一+1
ax1.set_ylabel('Accuracy (%)', fontsize=17)  # 16→17
ax1.set_ylim([25, 100])
ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.7)
ax1.set_xticks(x_axis)
ax1.set_xticklabels(subject_names, rotation=0, fontsize=16)  # 15→16
ax1.tick_params(axis='y', labelsize=16)  # 15→16

# 子图标注位置微调，字体+1
ax1.text(-2.5, 106, '(a)', fontsize=18, fontweight='bold')  # 17→18
ax1.set_title('CCS-HI Task', fontsize=18, fontweight='bold', pad=14)  # 17→18

# 图例字体+1
ax1.legend(loc='upper right', framealpha=0.95, fontsize=16)  # 15→16

# ====================== 4. 绘制(b) SV-HI ======================
ax2.fill_between(x_axis, baseline_min_sv, baseline_max_sv, color='#FFE0B2', alpha=0.5)
ax2.plot(x_axis, ours_sv, color='#FF8800', marker='o', markersize=8, lw=2.1)

# 所有字体统一+1
ax2.set_ylabel('Accuracy (%)', fontsize=17)  # 16→17
ax2.set_ylim([25, 100])
ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.7)
ax2.set_xticks(x_axis)
ax2.set_xticklabels(subject_names, rotation=0, fontsize=16)  # 15→16
ax2.tick_params(axis='y', labelsize=16)  # 15→16

# 子图标注位置微调，字体+1
ax2.text(-2.5, 106, '(b)', fontsize=18, fontweight='bold')  # 17→18
ax2.set_title('SV-HI Task', fontsize=18, fontweight='bold', pad=14)  # 17→18

# ====================== 5. 保存最终定稿图 ======================
plt.tight_layout()
# plt.savefig("Chinese_HI_Subject_Acc_Compare_final.pdf", format='pdf', facecolor='white')
plt.savefig("Chinese_HI_Subject_Acc_Compare_final.png", format='png', facecolor='white')
plt.show() """
import numpy as np
import matplotlib.pyplot as plt

# ====================== 1. ELL-HI 数据录入 ======================
subject_names = [f"S{i:02d}" for i in range(1, 15)] + ["Average"]
x_axis = np.arange(len(subject_names))

# 11个基线 + FineSTAN(最后一列)
data_ell = np.array([
    [44.00, 43.25, 42.25, 27.50, 44.00, 45.50, 44.75, 39.00, 43.50, 40.75, 41.25, 54.25],
    [35.75, 28.50, 27.00, 33.00, 34.00, 33.25, 32.50, 27.25, 37.75, 35.75, 36.75, 39.50],
    [64.50, 53.75, 58.75, 71.25, 64.50, 67.75, 70.50, 58.00, 66.75, 63.75, 69.25, 73.75],
    [44.75, 43.75, 54.00, 34.00, 47.75, 46.50, 56.00, 39.75, 55.75, 56.25, 54.50, 69.00],
    [37.25, 35.25, 37.25, 27.25, 32.00, 35.00, 37.25, 27.00, 39.50, 36.75, 31.50, 43.50],
    [27.75, 23.00, 32.00, 23.25, 32.25, 31.50, 31.75, 38.50, 25.55, 36.75, 32.00, 61.50],
    [80.50, 80.75, 79.50, 76.00, 72.00, 78.75, 79.75, 71.25, 82.50, 80.00, 80.00, 83.50],
    [71.75, 72.00, 60.75, 51.75, 80.50, 75.75, 74.50, 77.00, 72.50, 80.00, 72.00, 72.25],
    [52.50, 56.50, 57.50, 62.00, 65.50, 59.25, 63.75, 52.50, 61.00, 63.50, 57.25, 58.50],
    [52.25, 45.00, 53.50, 56.50, 54.50, 56.75, 51.75, 38.00, 58.75, 59.25, 52.25, 70.25],
    [28.00, 20.75, 34.75, 38.75, 30.50, 24.25, 27.25, 22.25, 29.75, 28.75, 30.25, 40.50],
    [55.75, 58.00, 60.50, 41.25, 54.50, 56.50, 59.50, 40.25, 63.75, 63.50, 56.75, 69.00],
    [33.50, 31.00, 35.50, 30.25, 33.50, 30.50, 33.00, 27.25, 31.25, 33.75, 33.25, 44.25],
    [54.50, 40.75, 59.25, 44.25, 58.50, 47.75, 59.50, 45.25, 59.50, 59.50, 50.75, 70.00]
])

baseline_ell = data_ell[:, :11]
ours_ell = data_ell[:, -1]
baseline_max_ell = np.max(baseline_ell, axis=1)
baseline_min_ell = np.min(baseline_ell, axis=1)

# 补充Average均值
baseline_max_ell = np.append(baseline_max_ell, np.mean(baseline_max_ell))
baseline_min_ell = np.append(baseline_min_ell, np.mean(baseline_min_ell))
ours_ell = np.append(ours_ell, np.mean(ours_ell))

# ====================== 2. 全局绘图参数【完全和之前CCS/SV统一，字号、字体不变】 ======================
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 16,
    'axes.linewidth': 1.4,
    'xtick.major.width': 1.4,
    'ytick.major.width': 1.4,
    'figure.dpi': 100,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
})

# 画布尺寸和原图保持一致
fig, ax = plt.subplots(1, 1, figsize=(16, 5))

# ====================== 3. 绘图配色、线条完全同源 ======================
ax.fill_between(x_axis, baseline_min_ell, baseline_max_ell, color='#FFE0B2', alpha=0.5, label='ACC Range (others)')
ax.plot(x_axis, ours_ell, color='#FF8800', marker='o', markersize=8, lw=2.1, label='FineSTAN ACC')

# Y轴标签字号、范围和之前完全一致
ax.set_ylabel('Accuracy (%)', fontsize=17)
ax.set_ylim([25, 100])
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.7)
ax.set_xticks(x_axis)
ax.set_xticklabels(subject_names, rotation=0, fontsize=16)
ax.tick_params(axis='y', labelsize=16)

# 左上角子图标注(a)，坐标、字号和原图一模一样
# ax.text(-1.8, 106, '(a)', fontsize=18, fontweight='bold')
ax.set_title('ELL-HI Task', fontsize=18, fontweight='bold', pad=14)

# 图例参数不变
ax.legend(loc='upper right', framealpha=0.95, fontsize=16)

# ====================== 4. 保存 ======================
# plt.savefig("ELL_HI_Subject_Compare.pdf", format='pdf', facecolor='white')
plt.savefig("ELL_HI_Subject_Compare.png", format='png', facecolor='white')
plt.show()