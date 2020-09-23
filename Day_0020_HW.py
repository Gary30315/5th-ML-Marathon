import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

import warnings
warnings.filterwarnings('ignore')

# matrix = np.random.random((10,10))
# plt.figure(figsize=(10,10))
# sns.heatmap(matrix, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
# plt.show()

# nrow = 1000
# ncol = 3

# matrix = np.random.random((1000,3))
# indice = np.random.choice([0,1,2], size=nrow)
# plot_data = pd.DataFrame(matrix, indice).reset_index()
# grid = sns.PairGrid(data = plot_data, size = 3, diag_sharey=False, 
#                     hue = 'index', vars = [x for x in list(plot_data.columns) if x != 'index'])
# grid.map_upper(plt.scatter, alpha = 0.2)
# grid.map_diag(plt.hist)
# grid.map_lower(sns.kdeplot, cmap = plt.cm.OrRd_r)
# plt.show()

nrow = 1000
ncol = 3

matrix = np.random.randn(1000,3)

# 隨機給予 0, 1, 2 三種標籤
indice = np.random.choice([0,1,2], size=nrow)
plot_data = pd.DataFrame(matrix, indice).reset_index()

# 繪製 seborn 進階 Heatmap
grid = sns.PairGrid(data = plot_data, size = 3, diag_sharey=False, 
                    hue = 'index', vars = [x for x in list(plot_data.columns) if x != 'index'])

grid.map_upper(plt.scatter, alpha = 0.2)
grid.map_diag(plt.hist)
grid.map_lower(sns.kdeplot,cmap = plt.cm.OrRd_r)

plt.show()