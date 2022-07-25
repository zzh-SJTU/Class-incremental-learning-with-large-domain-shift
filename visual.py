
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
x = np.array([0, 1, 2, 3, 4])
y_finetuning = np.array([99.63, 51.4, 33.72, 24.885, 20.882])
y_lwf = np.array([99.47, 51.295, 46.73, 36.26, 36.57])
y_ewc = np.array([99.63, 50.505, 36.663, 31.65, 26.642])
y_icarl = np.array([99.65, 97.88, 97.66, 96.41, 96.714])
y_finetuning_1 = np.array([95.4, 58.09, 30.52, 28.1125, 22.044])
y_lwf_1 = np.array([93.13, 56.79, 53.37, 44.47, 46.982])
y_ewc_1 = np.array([94.93, 49.52, 39.16, 18.2125, 12.89])
y_icarl_1 = np.array([99.65, 97.88, 97.65, 96.61, 96.714])
y_finetuning_2 = np.array([60.50, 37.15, 25.9, 23.15, 16.55])
y_lwf_2 = np.array([54.15, 37.45, 29.05, 25.4, 22.62])
y_ewc_2 = np.array([59.15, 41.875, 34.183, 29.8375, 27.68])
y_icarl_2 = np.array([53.45, 45.6, 40.35, 36.25, 33.4])
fig, axes = plt.subplots(1, 1, figsize=(6, 4))  
# first plot with X and Y data
axes.plot(x, y_finetuning_2, label = 'Finetuning',marker = '.')
axes.plot(x, y_lwf_2,color = 'm', label = 'LwF',marker = '.')
axes.plot(x, y_ewc_2,color = 'g', label = 'EWC',marker = '.')
axes.plot(x, y_icarl_2,color = 'y', label = 'icarl',marker = '.')
x1 = [2, 4, 6, 8]
y1 = [3, 5, 7, 9]
# second plot with x1 and y1 data
axes.set_yticks([10,20,30,40,50, 60, 70, 80, 90, 100])
axes.set_xticks([0,1,2,3,4])
axes.yaxis.set_minor_locator(MultipleLocator(2.5))
axes.xaxis.set_minor_locator(MultipleLocator(1))  
plt.legend(loc='upper right')
plt.xlabel("ACC after training task i")
plt.ylabel("ACC")
plt.title('Comparison between different methods on cifar100')
plt.show()
plt.savefig('line.jpg',dpi = 800)