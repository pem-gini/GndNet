from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt
import numpy as np

plt.ion()

def ff(x, y):
    return x**2 + y**2

heightmap = np.load('heightmap.npy')
heightmap_raw = np.load('heightmap_raw.npy')
num_points = np.load('num_points.npy')

variance_matrix = np.var(heightmap_raw, axis=2)

# hightmap_filtered_coords = np.load('axis.npy')
# hightmap_filtered = np.load('img.npy')


# Get all indeces with no values
y, x = np.where(num_points!=1)
xg, yg = np.meshgrid(x, y, indexing='ij')

hightmap_filtered = heightmap[y,x]
colors = num_points[y,x]
sizes = np.sqrt(variance_matrix[y,x]) * 100

interp = LinearNDInterpolator(list(zip(y,x)), hightmap_filtered)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
scatter = ax.scatter(y, x, hightmap_filtered, s=sizes, c=colors, label='data')

legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Num. points")
ax.add_artist(legend1)

kw = dict(prop="sizes", num=5, color=scatter.cmap(0.7),
          func=lambda s: (s / 100)**2)
legend2 = ax.legend(*scatter.legend_elements(**kw),
                    loc="lower right", title="Variance")#
ax.add_artist(legend2)

xx = np.arange(50)
yy = np.arange(50)
X, Y = np.meshgrid(xx, yy, indexing='ij')

# interpolator
ax.plot_wireframe(X, Y, interp((X, Y)), rstride=3, cstride=3, alpha=0.4, color='m', label='linear interp')

plt.legend()
plt.show(block=True)