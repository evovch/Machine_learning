import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import csg_constants as cnst

CANVAS_SIZE = 10 # inches

# -------------------------------
def get_bounding_box_vertices(s):
    return np.array([[ s,-s,-s],
                     [-s,-s,-s],
                     [-s, s,-s],
                     [ s, s,-s],
                     [ s,-s, s],
                     [-s,-s, s],
                     [-s, s, s],
                     [ s, s, s]])
# -------------------------------

# You can either supply a cloud of points of shape (n_points,3)
# [[xyz][xyz]...[xyz]]
# Or a feature vector of shape (n_points*3)
# [xyzxyz...xyz]
def draw_3d_cloud(fig_idx, pts, title):
    if (pts.ndim==1):
        pts = np.reshape(pts, (int(len(pts)/3),3))
    fig = plt.figure(fig_idx, figsize=(CANVAS_SIZE,CANVAS_SIZE)) # size in inches
    fig.suptitle(title)
    axs = fig.gca(projection='3d')
    bounding_box = get_bounding_box_vertices(cnst.SPACE_HALF_SIZE)
    axs.scatter(bounding_box[:,0], bounding_box[:,1], bounding_box[:,2], marker=',', s=(72./fig.dpi)**2)
    axs.scatter(pts[:,0], pts[:,1], pts[:,2], marker=',', s=(72./fig.dpi)**2)

# Input - full batch
def draw_3d_cloud_batch(pts, title_prefixes):
    for i in range(len(pts)):
        draw_3d_cloud(i, pts[i], title_prefixes[i] + str(i))

# You can either supply a cloud of points of shape (n_points,2)
# [[xy][xy]...[xy]]
# Or a feature vector of shape (n_points*2)
# [xyxy...xy]
def draw_2d_cloud(fig_idx, pts, title):
    if (pts.ndim==1):
        pts = np.reshape(pts, (int(len(pts)/2),2))
    fig = plt.figure(fig_idx, figsize=(CANVAS_SIZE,CANVAS_SIZE)) # size in inches
    fig.suptitle(title)
    axs = fig.gca()
    axs.grid(True)
    bounding_box = get_bounding_box_vertices(cnst.SPACE_HALF_SIZE)
    axs.scatter(bounding_box[:,0], bounding_box[:,1], marker=',', s=(72./fig.dpi)**2)
    axs.scatter(pts[:,0], pts[:,1], marker=',', s=(72./fig.dpi)**2)

# Input - full batch
def draw_2d_cloud_batch(pts, title_prefixes):
    for i in range(len(pts)):
        draw_2d_cloud(i, pts[i], title_prefixes[i] + str(i))
