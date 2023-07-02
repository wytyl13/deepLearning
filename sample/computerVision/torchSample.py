import utils
# from importlib import reload
# reload(utils)
import torch
import torch.nn as nn

from utils import set_default, show_scatterplot, plot_bases
from matplotlib.pyplot import plot, title, axis, figure, gca, gcf
import numpy as np
set_default()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

n_points = 1_000
X = torch.randn(n_points, 2).to(device)

x_min = -1.5
x_max = +1.5
colors = (X - x_min) / (x_max - x_min)
colors = (colors * 511).short().numpy()
colors = np.clip(colors, 0, 511)

figure().add_axes([0, 0, 1, 1])
show_scatterplot(X, colors, title = 'X')
OI = torch.cat((torch.zeros(2, 2), torch.eye(2))).to(device)
plot_bases(OI)
plot.show()