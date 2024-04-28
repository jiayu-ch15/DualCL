import matplotlib.pyplot as plt
fontsize = 22
def plot_multi_mean_std_graph(x_values_list, y_mean_values_list, y_std_values_list, x_label, y_label, labels, name, inverse=False, legend_loc='center right'):
    plt.style.use("ggplot")
    plt.figure(figsize=(5.4, 4.5))  # 设置画布大小
    for x_values, y_mean_values, y_std_values, label in zip(x_values_list, y_mean_values_list, y_std_values_list, labels):
        plt.errorbar(x_values, y_mean_values, yerr=y_std_values, fmt='-o', label=label)
    plt.xlabel(x_label, fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)
    plt.legend(loc=legend_loc, fontsize=15)
    plt.grid(True)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    if inverse:
        plt.gca().invert_xaxis()  # 反转 x 轴
    plt.tight_layout()
    plt.savefig('{}.pdf'.format(name))

# catch radius - capture rate
x_values_list = [[0.5, 0.4, 0.3, 0.2, 0.12], [0.5, 0.4, 0.3, 0.2, 0.12], [0.5, 0.4, 0.3, 0.2, 0.12], [0.5, 0.4, 0.3, 0.2, 0.12], [0.5, 0.4, 0.3, 0.2, 0.12], [0.5, 0.4, 0.3, 0.2, 0.12], [0.5, 0.4, 0.3, 0.2, 0.12]]
y_mean_values_list = [[1.0, 1.0, 1.0, 1.0, 0.988], [1.0, 0.85, 0.39, 0.043, 0.0], [1.0, 0.87, 0.447, 0.0333, 0.0], [1.0, 0.83, 0.417, 0.0333, 0.0], [1.0, 0.917, 0.433, 0.05, 0.0], [1.0, 0.320, 0.137, 0.028, 0.001], [1.0, 1.0, 1.0, 1.0, 0.924]]
y_std_values_list = [[0.0, 0.0, 0.0, 0.0, 0.004], [0.0, 0.008165, 0.008165, 0.004714, 0.0], [0.0, 0.01633, 0.012472, 0.0047, 0.0], [0.0, 0.008, 0.017, 0.005, 0.0], [0.0, 0.0125, 0.0125, 0.008, 0.0], [0.0, 0.006, 0.002, 0.003, 0.001], [0.0, 0.0, 0.0, 0.0, 0.018]]

# catch radius - capture timestep
y2_mean_values_list = [[32.33, 69.12, 151.21, 238.77, 353.03], [17.67, 150.67, 498.67, 764, 800.0], [17.67, 130.67, 495.33, 759.33, 800.0], [17, 148, 484.33, 745.33, 800], [17.67, 127.33, 489.33, 755.33, 800], [61.18, 567.00, 708.15, 778.81, 799.47], [32.51, 68.35, 156.12, 281.91, 439.36]]
y2_std_values_list = [[0.88, 3.46, 9.06, 7.79, 11.96], [0.471, 7.587, 6.128, 2.944, 0.0], [0.471, 3.682, 5.249, 4.989, 0.0], [0.816, 4.082, 8.731, 7.409, 0.0], [0.471, 6.944, 8.654, 4.190, 0.0], [5.76, 9.72, 4.21, 0.18, 0.377], [1.68, 4.09, 11.18, 3.37, 5.80]]
labels = ["DualCL", "Angelani", "Janosov", "APF", "DACOOP", "MAPPO", "MAPPO +\nIntrinsic"]

plot_multi_mean_std_graph(x_values_list, y_mean_values_list, y_std_values_list, "Catch Radius", "Capture Rate", labels, 'CatchRadius_rate', inverse=True, legend_loc='center right')
plot_multi_mean_std_graph(x_values_list, y2_mean_values_list, y2_std_values_list, "Catch Radius", "Capture Timestep", labels, 'CatchRadius_time', inverse=True, legend_loc='center left')

# speed - capture rate
x_values_list = [[1.2, 1.6, 2.0, 2.4], [1.2, 1.6, 2.0, 2.4], [1.2, 1.6, 2.0, 2.4], [1.2, 1.6, 2.0, 2.4],[1.2, 1.6, 2.0, 2.4],[1.2, 1.6, 2.0, 2.4], [1.2, 1.6, 2.0, 2.4]]
y_mean_values_list = [[0.993, 1.0, 1.0, 1.0,], [0.183, 0.05, 0.043, 0.043], [0.253, 0.053, 0.043, 0.033], [0.19, 0.123, 0.093, 0.033], [0.35, 0.157, 0.093, 0.05], [0.04, 0.036, 0.033, 0.001], [0.998, 0.994, 1.0, 1.0]]
y_std_values_list = [[0.004, 0.0, 0.0, 0.0,], [0.012, 0.008, 0.0047, 0.0047], [0.012, 0.0047, 0.0047, 0.0047], [0.0163, 0.0047, 0.02168, 0.0047], [0.02, 0.0125, 0.0125, 0.008], [0.004, 0.0, 0.0, 0.001], [0.0, 0.005, 0.0, 0.0]]

# speed - capture timestep
y2_mean_values_list = [[301.14, 252.12, 242.39, 239.33], [660.67, 765.67, 765.67, 764], [654.33, 762, 759.67, 759.33], [647, 744.67, 755.67, 745.33], [591.33, 722.33, 751.67, 755.33], [773.28, 771.52, 773.92, 799.47], [314.94, 321.69, 289.5, 281.91]]
y2_std_values_list = [[5.58, 5.97, 3.00, 8.49], [24.64, 3.30, 3.30, 2.94], [21.48, 5.72, 1.70, 4.99], [18.83, 10.53, 6.55, 7.41], [18.84, 8.73, 7.04, 4.19], [4.06, 0.372, 0.38, 0.377], [11.97, 16.32, 5.42, 3.37]]
labels = ["DualCL", "Angelani", "Janosov", "APF", "DACOOP", "MAPPO", "MAPPO +\nIntrinsic"]

plot_multi_mean_std_graph(x_values_list, y_mean_values_list, y_std_values_list, "Speed of Evader", "Capture Rate", labels, 'Speed_rate')
plot_multi_mean_std_graph(x_values_list, y2_mean_values_list, y2_std_values_list, "Speed of Evader", "Capture Timestep", labels, 'Speed_time')