import numpy as np
import random as rnd

import csg_constants as cnst

import matplotlib.pyplot as plt
from cloud_of_points import draw_2d_cloud

from csg_2d_shape import csg_2d_shape # mother class

# ==============================================================================

class csg_2d_generic_trapezoid(csg_2d_shape):

    def __init__(self):
        self._primitive_type = 3
        self._param_names = ['dx1','dx2','dy','alpha']
        self._params = [0., 0., 0., 0.]
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
                        rnd.uniform(cnst.MIN_LENGTH_VAL/2, cnst.SPACE_HALF_SIZE),
                        rnd.uniform(-90., 90)]
        self._constructed = True

    # Output point shape (2)
    # [xy]
    def gen_point_on_boundary(self):
        assert(self._constructed==True)
        dx1 = self._params[0]
        dx2 = self._params[1]
        dy  = self._params[2]
        alpha = self._params[3]
        face_id = self.get_random_face_id()
        if (face_id==0):
            y = rnd.uniform(-dy, dy)
            xc = y*np.tan(alpha*cnst.DEGTORAD)
            x = xc - (dx1 - (dy+y)*(dx1-dx2)/(2.*dy))
        elif (face_id==1):
            y = rnd.uniform(-dy, dy)
            xc = y*np.tan(alpha*cnst.DEGTORAD)
            x = xc + (dx1 - (dy+y)*(dx1-dx2)/(2.*dy))
        elif (face_id==2):
            y = -dy
            xc = y*np.tan(alpha*cnst.DEGTORAD)
            x = xc + rnd.uniform(-dx1, dx1)
        elif (face_id==3):
            y = dy
            xc = y*np.tan(alpha*cnst.DEGTORAD)
            x = xc + rnd.uniform(-dx2, dx2)
        return [x, y]

# ==============================================================================

if __name__ == "__main__":

    shp = csg_2d_generic_trapezoid()
    shp.gen_random()
    cloud = shp.gen_cloud_on_boundary(10000)

    draw_2d_cloud(1, cloud, "GENERIC_TRAPEZOID")

    plt.show()

    exit()
