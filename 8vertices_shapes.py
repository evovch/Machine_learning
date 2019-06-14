import random as rnd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -------------------------------------
def get_indices():
    return [[0,1], [1,3], [3,2], [2,0],
            [4,5], [5,7], [7,6], [6,4],
            [0,4], [1,5], [3,7], [2,6]]
# -------------------------------------

# ------------------------------------------------------------------------
def draw_primitive(fig_idx, vtc):
    fig1 = plt.figure(fig_idx, figsize=(10,10)) # size in inches
    fig1.suptitle('Primitive')
    axs1 = fig1.gca(projection='3d')
    edge_p = np.linspace(0., 1., 50)
    indices = get_indices()
    for idx in indices:
        indexA = idx[0]
        indexB = idx[1]
        axs1.plot(vtc[indexA][0] + (vtc[indexB][0]-vtc[indexA][0])*edge_p,
                  vtc[indexA][1] + (vtc[indexB][1]-vtc[indexA][1])*edge_p,
                  vtc[indexA][2] + (vtc[indexB][2]-vtc[indexA][2])*edge_p)
# ------------------------------------------------------------------------

# --------------------------------------------------
# argument: list ['dx', 'dy', 'dz']
# return: [8][3] array of vertices coordinates
def build_BOX(params):
    p_dx = params[0]
    p_dy = params[1]
    p_dz = params[2]
    vtc = [[ p_dx,-p_dy,-p_dz], [-p_dx,-p_dy,-p_dz],
           [ p_dx, p_dy,-p_dz], [-p_dx, p_dy,-p_dz],
           [ p_dx,-p_dy, p_dz], [-p_dx,-p_dy, p_dz],
           [ p_dx, p_dy, p_dz], [-p_dx, p_dy, p_dz]]
    return vtc
# --------------------------------------------------

# ----------------------------------------------------------
# argument: list ['dx', 'dy', 'dz', 'alpha', 'theta', 'phi']
# return: [8][3] array of vertices coordinates
# angles in degrees
# def build_PARA(params):
#     p_dx = params[0]
#     p_dy = params[1]
#     p_dz = params[2]
#     p_alpha = params[3]
#     p_theta = params[4]
#     p_phi = params[5]
#     vtc = []
#     return vtc
# ----------------------------------------------------------

# ------------------------------------------------------
# argument: list ['dx1', 'dy1', 'dx2', 'dy2', 'dz']
# return: [8][3] array of vertices coordinates
def build_TRD(params):
    p_dx1 = params[0]
    p_dy1 = params[1]
    p_dx2 = params[2]
    p_dy2 = params[3]
    p_dz  = params[4]
    vtc = [[ p_dx1,-p_dy1,-p_dz], [-p_dx1,-p_dy1,-p_dz],
           [ p_dx1, p_dy1,-p_dz], [-p_dx1, p_dy1,-p_dz],
           [ p_dx2,-p_dy2, p_dz], [-p_dx2,-p_dy2, p_dz],
           [ p_dx2, p_dy2, p_dz], [-p_dx2, p_dy2, p_dz]]
    return vtc
# ------------------------------------------------------

# ---------------------------------------------------------
# argument: list ['dz', 'theta', 'phi',
# 'dy1', 'dx1', 'dx2', 'alpha1', 'dy2', 'dx3', 'dx4']
# return: [8][3] array of vertices coordinates
# angles in degrees
# Note that alpha2 = alpha1, so it is not provided as input
# def build_TRAP(params):
#     p_dz = params[0]
#     p_theta = params[1]
#     p_phi = params[2]
#     p_dy1 = params[3]
#     p_dx1  = params[4]
#     p_dx2  = params[5]
#     p_alpha1  = params[6]
#     p_dy2 = params[7]
#     p_dx3  = params[8]
#     p_dx4  = params[9]
#     vtc = []
#     return vtc
# ---------------------------------------------------------

# -----------------------------------------------
def gen_random_box(maxCoord):
    return build_BOX([rnd.uniform(0., maxCoord),
                      rnd.uniform(0., maxCoord),
                      rnd.uniform(0., maxCoord)])
# -----------------------------------------------

# -----------------------------------------------
# def gen_random_para(maxCoord):
#     return build_PARA()
# -----------------------------------------------

# -----------------------------------------------
def gen_random_trd(maxCoord):
    return build_TRD([rnd.uniform(0., maxCoord),
                      rnd.uniform(0., maxCoord),
                      rnd.uniform(0., maxCoord),
                      rnd.uniform(0., maxCoord),
                      rnd.uniform(0., maxCoord)])
# -----------------------------------------------

# -----------------------------------------------
# def gen_random_trap(maxCoord):
#     return build_TRAP()
# -----------------------------------------------

# box_params = [3., 5., 11.]
# box_vtc = build_BOX(box_params)
box_vtc = gen_random_box(100.)
draw_primitive(1, box_vtc)

# para_params = [3., 5., 11., 10., 20., 30.]
# para_vtc = build_PARA(para_params)
# draw_primitive(2, para_vtc)

# trd_params = [10., 20., 5., 8., 10.]
# trd_vtc = build_TRD(trd_params)
trd_vtc = gen_random_trd(100.)
draw_primitive(3, trd_vtc)

# trap_params = [10., 0., 0., 5., 6., 7., 0., 11., 12., 13.]
# trap_vtc = build_TRAP(trap_params)
# draw_primitive(4, trap_vtc)

# This should be called only once in the very end
plt.show()
