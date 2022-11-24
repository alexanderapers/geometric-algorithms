import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

n_points = 20

def find_largest_triangle(red_point, pointcloud):
    A = np.append(red_point, 1)
    B = np.append(pointcloud, np.ones(pointcloud.shape[0]).reshape(-1, 1), axis=1)

    max = 0
    solution = (0, 0)

    for i in range(B.shape[0]):
        for j in range(B.shape[0]):
            C = np.stack([A, B[i], B[j]])
            area = np.linalg.det(C) / 2 
            if area > max:
                max = area
                solution = (i, j)

    return np.stack([A, B[solution[0]], B[solution[1]]])[:, 0:2]

pointcloud = np.random.rand(n_points, 2)
red_point = np.random.rand(1, 2)
ch = ConvexHull(np.concatenate([pointcloud, red_point]))

plt.scatter(x=pointcloud[:,0], y=pointcloud[:,1])
plt.scatter(x=red_point[:,0], y=red_point[:,1], color="red")
for simplex in ch.simplices:
    plt.plot(ch.points[simplex, 0], ch.points[simplex, 1], "r--")

result = find_largest_triangle(red_point, pointcloud)
plt.plot(result[:, 0], result[:, 1], "purple")
plt.plot([result[-1, 0], result[0, 0]], [result[-1, 1], result[0, 1]], "purple")


ax = plt.gca()
ax.set_aspect('equal', adjustable="box")
plt.savefig("figure_1.png")

