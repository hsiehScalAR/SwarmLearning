import numpy as np
import csv
from matplotlib import pyplot as plt
from matplotlib import animation
import sys
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
import visualize_conn as vc
# swarm_square_agent100_a5_conn20_it2000.csv

my_data = []
with open(sys.argv[1]) as csvDataFile:
	csvReader = csv.reader(csvDataFile)
	for row in csvReader:
		my_data.append(row)

#my_data = np.genfromtxt('swarm_ring_500_a1.csv', delimiter=',')
pos_from_file = []
vel_from_file = []
line_pos = []
line_vel = []
for i in range(len(my_data)):
	for j in range(len(my_data[0])/2):
		line_str = my_data[i][j*2].lstrip('[').rstrip(']').split()
		line_pos.append([float(line_str[0]), float(line_str[1])])
		line_str = my_data[i][j*2+1].lstrip('[').rstrip(']').split()
		line_vel.append([float(line_str[0]), float(line_str[1])])
	pos_from_file.append(line_pos)
	vel_from_file.append(line_vel)
	line_vel = []
	line_pos = []

vel_from_file = np.array(vel_from_file)
pos_from_file = np.array(pos_from_file)



J = []

for j in range(100):
	vel_dot = []
	f_vel = []
	#print vel_from_file.shape
	for i in range(len(vel_from_file)-1):
		vel_dot.append(np.array(vel_from_file[i+1][j]-vel_from_file[i][j])/0.00005)
	for i in range(len(vel_from_file)-1):
		f_vel.append((1-np.dot(vel_from_file[i][j],vel_from_file[i][j]))*vel_from_file[i][j])

	vel_dot = np.array(vel_dot)
	f_vel = np.array(f_vel)

	X = np.transpose(vel_dot+f_vel)
	G = -5.0/20*np.tile(pos_from_file[1:,j,:],(pos_from_file.shape[1],1,1))-np.swapaxes(pos_from_file[1:,:,:],0,1)
	X = X.reshape(X.shape[1]*2)
	G = G.reshape(G.shape[0],G.shape[1]*2)
	G = np.transpose(G)
	#print X.shape
	#print G.shape
	'''
	# G is not invertible, try linear regression instead
	GGT = np.matmul(G,np.transpose(G))
	np.matmul(X, np.matmul(np.transpose(G), np.linalg.inv(GGT)))
	'''
	J_lstsq = np.linalg.lstsq(G,X)[0]
	Q,R = np.linalg.qr(G)
	Qb = np.dot(Q.T,X)
	J_qr = np.linalg.solve(R,Qb)
	J.append(J_qr)

J = np.array(J)
print J

def heatmap(data, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=0.1)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

fig, ax = plt.subplots()

im, cbar = heatmap(J.T,ax=ax,
                   cmap="YlGn", cbarlabel="Connectivity Strength")

ax.set_title("Swarm Connectivity")
fig.tight_layout()
plt.show()