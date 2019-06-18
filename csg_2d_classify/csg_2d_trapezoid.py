import numpy as np
import random as rnd

import csg_constants as cnst

import matplotlib.pyplot as plt
from cloud_of_points import draw_2d_cloud

from csg_2d_shape import csg_2d_shape # mother class

# ==============================================================================

class csg_2d_trapezoid(csg_2d_shape):

    def __init__(self):
        self._primitive_type = 2
        self._param_names = ['dx1','dx2','dy']
        self._params = [0., 0., 0.]
        self._face_names = ['min_x','max_x','min_y','max_y']
        self._constructed = False

    def get_subtype(self):
        assert(self._constructed==True)
        return 0 # min_x, max_x, min_y, max_y

    def get_faces(self):
        assert(self._constructed==True)
        subtype = self.get_subtype()
        if (subtype == 0):
            return [0, 1, 2, 3] # min_x, max_x, min_y, max_y

    def gen_random(self):
        self._params = [rnd.uniform(cnst.MIN_LENGTH_VAL/2, cnst.SPACE_HALF_SIZE),
                        rnd.uniform(cnst.MIN_LENGTH_VAL/2, cnst.SPACE_HALF_SIZE),
                        rnd.uniform(cnst.MIN_LENGTH_VAL/2, cnst.SPACE_HALF_SIZE)]
        self._constructed = True

    # TODO check, implement correct boundary thickness
    # # Output point shape (2)
    # # [xy]
    # def gen_point_inside(self):
    #     assert(self._constructed==True)
    #     dx1 = self._params[0]
    #     dx2 = self._params[1]
    #     dy  = self._params[2]
    #     bth = cnst.BOUNDARY_THICKNESS/2 # boundary thickness half
    #     y = rnd.uniform(-dy+bth, dy-bth)
    #     xmax = dx1 - (dy+y)*(dx1-dx2)/(2.*dy)
    #     x = rnd.uniform(-xmax, xmax)
    #     return [x, y]

    # Output point shape (2)
    # [xy]
    def gen_point_on_boundary(self):
        assert(self._constructed==True)
        dx1 = self._params[0]
        dx2 = self._params[1]
        dy  = self._params[2]
        face_id = self.get_random_face_id()
        if (face_id==0):
            y = rnd.uniform(-dy, dy)
            x = -(dx1 - (dy+y)*(dx1-dx2)/(2.*dy))
            return [x, y]
        elif (face_id==1):
            y = rnd.uniform(-dy, dy)
            x = dx1 - (dy+y)*(dx1-dx2)/(2.*dy)
            return [x, y]
        elif (face_id==2):
            return [rnd.uniform(-dx1, dx1), -dy]
        elif (face_id==3):
            return [rnd.uniform(-dx2, dx2),  dy]

# ==============================================================================

if __name__ == "__main__":

    shp = csg_2d_trapezoid()
    shp.gen_random()
    cloud = shp.gen_cloud_on_boundary(10000)

    draw_2d_cloud(1, cloud, "TRAPEZOID")

    plt.show()

    exit()
