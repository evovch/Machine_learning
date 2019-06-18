import numpy as np
import random as rnd

import csg_constants as cnst

import matplotlib.pyplot as plt
from cloud_of_points import draw_2d_cloud

from csg_2d_shape import csg_2d_shape # mother class

# ==============================================================================

class csg_2d_full_ellipse(csg_2d_shape):

    def __init__(self):
        self._primitive_type = 6
        self._param_names = ['dx','dy']
        self._params = [0., 0.]
        self._face_names = ['outer_arc']
        self._constructed = False

    def get_subtype(self):
        assert(self._constructed==True)
        return 0 # outer_arc

    def get_faces(self):
        assert(self._constructed==True)
        subtype = self.get_subtype()
        if (subtype == 0):
            return [0] # outer_arc

    def gen_random(self):
        self._params = [rnd.uniform(cnst.MIN_LENGTH_VAL/2, cnst.SPACE_HALF_SIZE),
                        rnd.uniform(cnst.MIN_LENGTH_VAL/2, cnst.SPACE_HALF_SIZE)]
        self._constructed = True

    # Output point shape (2)
    # [xy]
    def gen_point_on_boundary(self):
        assert(self._constructed==True)
        a = self._params[0]
        b = self._params[1]
        face_id = self.get_random_face_id()
        if (face_id==0):
            phi = rnd.uniform(0., 360.)
            x = a * np.cos(phi*cnst.DEGTORAD)
            y = b * np.sin(phi*cnst.DEGTORAD)
            return [x, y]

# ==============================================================================

if __name__ == "__main__":

    shp = csg_2d_full_ellipse()
    shp.gen_random()
    cloud = shp.gen_cloud_on_boundary(10000)

    draw_2d_cloud(1, cloud, "FULL_ELLIPSE")

    plt.show()

    exit()
