import numpy as np
import random as rnd

import csg_constants as cnst

import matplotlib.pyplot as plt
from cloud_of_points import draw_cloud

# ==============================================================================
# TUBS
# ==============================================================================

class csg_tubs():

    def __init__(self):
        self._param_names = ['rmin','rmax','dz','sphi','dphi']
        self._params = [0., 0., 0., 0., 0.]
        self._face_names = ['bottom','top','outer_cyl','inner_cyl','planar_min_phi','planar_max_phi']
        self._constructed = False

    def get_subtype(self):
        assert(self._constructed==True)
        rmin = self._params[0]
        dphi = self._params[4]
        if (rmin < cnst.LENGTH_TOLERANCE and dphi > 360.-cnst.ANGLE_TOLERANCE):
            return 0 # bottom, top, outer_cyl
        elif (rmin < cnst.LENGTH_TOLERANCE and dphi <= 360.-cnst.ANGLE_TOLERANCE):
            return 1 # bottom, top, outer_cyl, planar_min_phi, planar_max_phi
        elif (rmin > cnst.LENGTH_TOLERANCE and dphi > 360.-cnst.ANGLE_TOLERANCE):
            return 2 # bottom, top, outer_cyl, inner_cyl
        elif (rmin >= cnst.LENGTH_TOLERANCE and dphi <= 360.-cnst.ANGLE_TOLERANCE):
            return 3 # bottom, top, outer_cyl, inner_cyl, planar_min_phi, planar_max_phi

    def get_faces(self):
        assert(self._constructed==True)
        subtype = self.get_subtype()
        if (subtype == 0):
            return [0, 1, 2]          # bottom, top, outer_cyl
        elif (subtype == 1):
            return [0, 1, 2, 4, 5]    # bottom, top, outer_cyl, planar_min_phi, planar_max_phi
        elif (subtype == 2):
            return [0, 1, 2, 3]       # bottom, top, outer_cyl, inner_cyl
        elif (subtype == 3):
            return [0, 1, 2, 3, 4, 5] # bottom, top, outer_cyl, inner_cyl, planar_min_phi, planar_max_phi

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
        rmin = rnd.uniform(0., cnst.SPACE_HALF_SIZE)
        rmax = rnd.uniform(rmin+cnst.MIN_LENGTH_VAL/2, cnst.SPACE_HALF_SIZE+cnst.MIN_LENGTH_VAL/2)
        self._params = [rmin,
                        rmax,
                        rnd.uniform(cnst.MIN_LENGTH_VAL/2, cnst.SPACE_HALF_SIZE),
                        rnd.uniform(0., 360.),
                        rnd.uniform(cnst.MIN_ANGLE_VAL, 360.)]
        self._constructed = True

    def gen_random_of_subtype(self, subtype):
        if (subtype == 0):
            self.gen_random()
            self._params[0] = 0.
            self._params[4] = 360.
        elif (subtype == 1):
            self.gen_random()
            self._params[0] = 0.
        elif (subtype == 2):
            self.gen_random()
            self._params[4] = 360.
        elif (subtype == 3):
            self.gen_random()

    # Output point shape (3)
    # [xyz]
    # TODO implement boundary for the planar faces
    def gen_point_inside(self):
        assert(self._constructed==True)
        rmin = self._params[0]
        rmax = self._params[1]
        dz   = self._params[2]
        sphi = self._params[3]
        dphi = self._params[4]
        bth = cnst.BOUNDARY_THICKNESS/2 # boundary thickness half
        r = rnd.uniform(rmin+bth, rmax-bth)
        phi = rnd.uniform(sphi, sphi+dphi)
        z = rnd.uniform(-dz+bth, dz-bth)
        x = r * np.cos(phi*cnst.DEGTORAD)
        y = r * np.sin(phi*cnst.DEGTORAD)
        return [x, y, z]

    # Output point shape (3)
    # [xyz]
    def gen_point_on_boundary(self):
        assert(self._constructed==True)
        rmin = self._params[0]
        rmax = self._params[1]
        dz   = self._params[2]
        sphi = self._params[3]
        dphi = self._params[4]
        face_id = self.get_random_face_id()
        if (face_id == 0): # bottom
            r = rnd.uniform(rmin, rmax)
            phi = rnd.uniform(sphi, sphi+dphi)
            z = -dz
        elif (face_id == 1): # top
            r = rnd.uniform(rmin, rmax)
            phi = rnd.uniform(sphi, sphi+dphi)
            z = dz
        elif (face_id == 2): # outer_cyl
            r = rmax
            phi = rnd.uniform(sphi, sphi+dphi)
            z = rnd.uniform(-dz, dz)
        elif (face_id == 3): # inner_cyl
            r = rmin
            phi = rnd.uniform(sphi, sphi+dphi)
            z = rnd.uniform(-dz, dz)
        elif (face_id == 4): # planar_min_phi
            r = rnd.uniform(rmin, rmax)
            phi = sphi
            z = rnd.uniform(-dz, dz)
        elif (face_id == 5): # planar_max_phi
            r = rnd.uniform(rmin, rmax)
            phi = sphi+dphi
            z = rnd.uniform(-dz, dz)
        x = r * np.cos(phi*cnst.DEGTORAD)
        y = r * np.sin(phi*cnst.DEGTORAD)
        return [x, y, z]

    # Output cloud shape (n_points,3):
    # [[xyz][xzy]...[xzy]]
    def gen_cloud_inside(self, n_points):
        assert(self._constructed==True)
        cloud = np.zeros((n_points,3))
        for i in range(n_points):
            cloud[i] = self.gen_point_inside()
        return cloud

    # Output cloud shape (n_points,3):
    # [[xyz][xzy]...[xzy]]
    def gen_cloud_on_boundary(self, n_points):
        assert(self._constructed==True)
        cloud = np.zeros((n_points,3))
        for i in range(n_points):
            cloud[i] = self.gen_point_on_boundary()
        return cloud

# ==============================================================================

if __name__ == "__main__":

    shp = csg_tubs()
    shp.gen_random()
    cloud = shp.gen_cloud_on_boundary(10000)

    draw_cloud(1, cloud, "TUBS")
    plt.show()

    exit()
