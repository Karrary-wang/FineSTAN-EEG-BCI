import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
import os  # 解决桌面保存权限问题

# 数据（完全保留）
ct_ccs = np.array([
    [0.6132, 0.0529, 0.0512, 0.2108, 0.0719],
    [0.0432, 0.6378, 0.2321, 0.0488, 0.0381],
    [0.0289, 0.1997, 0.6536, 0.0515, 0.0664],
    [0.1193, 0.0875, 0.0777, 0.6179, 0.0976],
    [0.0810, 0.0500, 0.0727, 0.1119, 0.6845]
])
ct_sv = np.array([
    [0.5258, 0.1888, 0.0705, 0.0563, 0.0962, 0.0624],
    [0.1746, 0.5513, 0.0579, 0.0810, 0.0792, 0.0561],
    [0.0514, 0.0741, 0.7589, 0.0370, 0.0393, 0.0393],
    [0.0503, 0.0773, 0.0371, 0.6830, 0.0814, 0.0709],
    [0.1108, 0.0779, 0.0479, 0.1193, 0.4821, 0.1620],
    [0.0393, 0.1018, 0.0529, 0.1192, 0.1385, 0.5482]
])

sat_ccs = np.array([
    [0.5913, 0.0550, 0.0619, 0.2234, 0.0684],
    [0.0432, 0.6301, 0.2036, 0.0658, 0.0574],
    [0.0345, 0.1872, 0.6729, 0.0467, 0.0586],
    [0.1324, 0.0714, 0.0616, 0.6327, 0.1018],
    [0.0714, 0.0324, 0.0734, 0.1050, 0.7177]
])
sat_sv = np.array([
    [0.5229, 0.1790, 0.0625, 0.0477, 0.1426, 0.0453],
    [0.1727, 0.5725, 0.0507, 0.0787, 0.0730, 0.0525],
    [0.0597, 0.0693, 0.7420, 0.0430, 0.0406, 0.0454],
    [0.0430, 0.0778, 0.0448, 0.6976, 0.0844, 0.0525],
    [0.0946, 0.0692, 0.0710, 0.0989, 0.5222, 0.1441],
    [0.0563, 0.0600, 0.0661, 0.0943, 0.1575, 0.5657]
])

fine_ccs = np.array([
    [0.6843, 0.0540, 0.0468, 0.1585, 0.0564],
    [0.0204, 0.7584, 0.1599, 0.0325, 0.0288],
    [0.0276, 0.2115, 0.6983, 0.0192, 0.0433],
    [0.0913, 0.0853, 0.0409, 0.7248, 0.0577],
    [0.0708, 0.0444, 0.0792, 0.1020, 0.7035]
])
fine_sv = np.array([
    [0.5479, 0.1661, 0.0585, 0.0526, 0.1147, 0.0603],
    [0.1147, 0.6320, 0.0430, 0.0800, 0.0812, 0.0490],
    [0.0537, 0.0489, 0.7936, 0.0310, 0.0394, 0.0334],
    [0.0299, 0.0585, 0.0299, 0.7706, 0.0585, 0.0526],
    [0.1050, 0.0621, 0.0525, 0.0967, 0.5430, 0.1408],
    [0.0239, 0.0585, 0.0525, 0.0788, 0.0979, 0.6885]
])

label_ccs = ['一', '丨', '丿', '㇏', 'ㄥ'] # ['一', '丨', '丿', '㇏', '𠃋']
label_sv = ['a', 'o', 'e', 'i', 'u', 'ü']
model_names = ['CTNet [27]', 'SATrans-Net [28]', 'FineSTAN (Ours)']
data_all = [[ct_ccs, sat_ccs, fine_ccs], [ct_sv, sat_sv, fine_sv]]
label_all = [label_ccs, label_sv]

# 字体修复（解决字符缺失警告，保留原有设置）
font_arial = FontProperties(family='Arial')
font_cn = FontProperties(family= 'Microsoft YaHei')  # 替换SimHei，支持所有汉字 'Microsoft YaHei'

# 字号设置（完全保留）
anno_fontsize = 12
tick_cn_size = 12
tick_en_size = 13
title_fontsize = 12

# 绘图函数（完全保留你的所有设置）
def plot_single_cm(ax, cm_raw, tick_labels, title_str, is_chinese, show_yaxis=False, row_idx=0, col_idx=0):
    cm_percent = np.round(cm_raw * 100, 2)
    sns.heatmap(
        cm_percent, ax=ax, annot=True, fmt='.2f',
        annot_kws={"size": anno_fontsize, "fontproperties": font_arial},
        cmap='Blues', vmin=0, vmax=80, cbar=False, square=True
    )

    ax.set_xticklabels(tick_labels, rotation=0)
    if show_yaxis:
        ax.set_yticklabels(tick_labels, rotation=0)
    else:
        ax.set_yticks([])

    ax.set_title(title_str, fontsize=title_fontsize, fontweight='bold', family='Arial')
    ax.set_xlabel('Predicted labels', fontsize=13, fontproperties=font_arial)

    if show_yaxis:
        ax.set_ylabel('True labels', fontsize=13, fontproperties=font_arial)

    xt = ax.get_xticklabels()
    yt = ax.get_yticklabels()
    if is_chinese:
        for xl in xt:
            xl.set_fontproperties(font_cn)
            xl.set_fontsize(tick_cn_size)
        for yl in yt:
            yl.set_fontproperties(font_cn)
            yl.set_fontsize(tick_cn_size)
    else:
        for xl in xt:
            xl.set_fontproperties(font_arial)
            xl.set_fontsize(tick_en_size)
        for yl in yt:
            yl.set_fontproperties(font_arial)
            yl.set_fontsize(tick_en_size)

    if col_idx == 0:
        tag = '(a)' if row_idx == 0 else '(b)'
        # 保留你调整好的位置：-0.15，字号14
        ax.text(-0.15, 1.04, tag, transform=ax.transAxes, fontsize=14, fontweight='bold', fontproperties=font_arial)

# 画布设置（完全保留）
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(13, 9))
plt.subplots_adjust(wspace=0.08, hspace=0.25, left=0.05, right=0.88, top=0.95, bottom=0.05)

# 绘制子图（完全保留）
for row_idx in range(2):
    lab_list = label_all[row_idx]
    for col_idx in range(3):
        ax = axes[row_idx, col_idx]
        data = data_all[row_idx][col_idx]
        show_y = (col_idx == 0)
        plot_single_cm(ax, data, lab_list, model_names[col_idx], is_chinese=(row_idx == 0), show_yaxis=show_y,
                       row_idx=row_idx, col_idx=col_idx)

# 色条（完全保留，无标题）
cbar_ax1 = fig.add_axes([0.90, 0.56, 0.013, 0.38])
cbar_ax2 = fig.add_axes([0.90, 0.06, 0.013, 0.38])

cb1 = fig.colorbar(axes[0, 0].collections[0], cax=cbar_ax1)
cb1.ax.tick_params(labelsize=11)
for t in cb1.ax.get_yticklabels():
    t.set_fontproperties(font_arial)

cb2 = fig.colorbar(axes[1, 0].collections[0], cax=cbar_ax2)
cb2.ax.tick_params(labelsize=11)
for t in cb2.ax.get_yticklabels():
    t.set_fontproperties(font_arial)

# ===================== 核心修复：高清保存到桌面，无权限报错 =====================
save_path = r"D:\EEG_HandWriting_Project_Code\SPD-HandWriteFormer\Figure6_Chinese_CM.png"
plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0)
# ================================================================================

plt.show()