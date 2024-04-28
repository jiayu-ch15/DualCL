import matplotlib.pyplot as plt
import numpy as np
fontsize = 22
def plot_multi_bar_graph(x_values, y_values_list, y_std_values_list, x_label, y_label, labels):
    plt.style.use("ggplot")
    plt.figure(figsize=(5.4, 4.5))  # 设置画布大小
    x = np.arange(len(x_values))
    width = 0.2
    for i, (y_values, y_std_values, label) in enumerate(zip(y_values_list, y_std_values_list, labels)):
        plt.bar(x + i*width, y_values, width=width, label=label)
        for j, (x_val, y_val, y_std) in enumerate(zip(x_values, y_values, y_std_values)):
            plt.vlines(x_val + i*width, y_val - y_std, y_val + y_std, colors='black')

    plt.xlabel(x_label, fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)
    plt.xticks(x + width * (len(y_values_list) - 1) / 2, x_values)
    plt.legend(loc='center left', fontsize=15)
    plt.grid(True)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig('bar.pdf')

# 示例数据
x_values = [0, 1, 2, 3]
y_values_list = [
    [0.003, 0.004, 0.006, 0.008],  # Algorithm 1
    [0.001, 0.003, 0.003, 0.005],  # Algorithm 2
    [0.937, 0.843, 0.719, 0.649],  # Algorithm 3
    [0.989, 0.976, 0.962, 0.965]   # Algorithm 4
]
y_std_values_list = [
    [0.005, 0.004, 0.003, 0.006],  # Algorithm 1
    [0.001, 0.002, 0.002, 0.001],  # Algorithm 2
    [0.025, 0.017, 0.017, 0.025],  # Algorithm 3
    [0.002, 0.004, 0.001, 0.001]   # Algorithm 4
]
labels = ["MAPPO", "MAPPO + External", "MAPPO + Intrinsic", "DualCL"]

# 设置图形属性并画图
plot_multi_bar_graph(x_values, y_values_list, y_std_values_list, "Number of Obstacles", "Capture Rate", labels)
