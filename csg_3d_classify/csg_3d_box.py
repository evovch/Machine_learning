import numpy as np
import random as rnd

import csg_constants as cnst

import matplotlib.pyplot as plt
from cloud_of_points import draw_cloud

# ==============================================================================
# BOX
# ==============================================================================

class csg_box():

    def __init__(self):
        self._param_names = ['dx','dy','dz']
        self._params = [0., 0., 0.]
        self._face_names = ['bottom','top','min_x','max_x','min_y','max_y']
        self._constructed = False

    def get_subtype(self):
        assert(self._constructed==True)
        return 0 # bottom, top, min_x, max_x, min_y, max_y

    def get_faces(self):
        assert(self._constructed==True)
        subtype = self.get_subtype()
        if (subtype == 0):
            return [0, 1, 2, 3, 4, 5] # bottom, top, min_x, max_x, min_y, max_y

    def get_n_faces(self):
        assert(self._constructed==True)
        faces = self.get_faces()
        return len(faces)

    def get_random_face_id(self):
        assert(self._constructed==True)
        faces = self.get_faces()
        idx = rnd.choice(range(self.get_n_faces()))
        return faces[idx]

    def gen_random(self):
        self._params = [rnd.uniform(cnst.MIN_LENGTH_VAL/2, cnst.SPACE_HALF_SIZE),
                        rnd.uniform(cnst.MIN_LENGTH_VAL/2, cnst.SPACE_HALF_SIZE),
                        rnd.uniform(cnst.MIN_LENGTH_VAL/2, cnst.SPACE_HALF_SIZE)]
        self._constructed = True


    # Output point shape (3)
    # [xyz]
    def gen_point_inside(self):
        assert(self._constructed==True)
        dx = self._params[0]
        dy = self._params[1]
        dz = self._params[2]
        bth = cnst.BOUNDARY_THICKNESS/2 # boundary thickness half
        return [rnd.uniform(-dx+bth, dx-bth),
                rnd.uniform(-dy+bth, dy-bth),
                rnd.uniform(-dz+bth, dz-bth)]

    # Output point shape (3)
    # [xyz]
    def gen_point_on_boundary(self):
        assert(self._constructed==True)
        dx = self._params[0]
        dy = self._params[1]
        dz = self._params[2]
        face_id = self.get_random_face_id()
        if (face_id==0):
            return [rnd.uniform(-dx, dx), rnd.uniform(-dy, dy), -dz]
        elif (face_id==1):
            return [rnd.uniform(-dx, dx), rnd.uniform(-dy, dy),  dz]
        elif (face_id==2):
            return [rnd.uniform(-dx, dx), -dy, rnd.uniform(-dz, dz)]
        elif (face_id==3):
            return [rnd.uniform(-dx, dx),  dy, rnd.uniform(-dz, dz)]
        elif (face_id==4):
            return [-dx, rnd.uniform(-dy, dy), rnd.uniform(-dz, dz)]
        elif (face_id==5):
            return [ dx, rnd.uniform(-dy, dy), rnd.uniform(-dz, dz)]

    # Output cloud shape (n_points,3):
    # [[xyz][xyz]...[xyz]]
    def gen_cloud_inside(self, n_points):
        assert(self._constructed==True)
        cloud = np.zeros((n_points,3))
        for i in range(n_points):
            cloud[i] = self.gen_point_inside()
        return cloud

    # Output cloud shape (n_points,3):
    # [[xyz][xyz]...[xyz]]
    def gen_cloud_on_boundary(self, n_points):
        assert(self._constructed==True)
        cloud = np.zeros((n_points,3))
        for i in range(n_points):
            cloud[i] = self.gen_point_on_boundary()
        return cloud

# ==============================================================================

if __name__ == "__main__":

    shp = csg_box()
    shp.gen_random()
    cloud = shp.gen_cloud_on_boundary(10000)

    draw_cloud(1, cloud, "BOX")
    plt.show()

    exit()
