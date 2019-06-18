import numpy as np
import random as rnd

import csg_constants as cnst

import matplotlib.pyplot as plt
from cloud_of_points import draw_cloud

# ==============================================================================
# SPHERE
# ==============================================================================

class csg_sphere():
    
    def __init__(self):
        self._param_names = ['rmin','rmax','sphi','dphi','stheta','dtheta']
        self._params = [0., 0., 0., 0., 0., 0.]
        self._face_names = ['outer_sph','inner_sph','planar_min_phi','planar_max_phi','conical_min_theta','conical_max_theta']
        self._constructed = False

    def get_subtype(self):
        assert(self._constructed==True)
        rmin = self._params[0]
        dphi = self._params[3]
        stheta = self._params[4]
        dtheta = self._params[5]

        flag1 = (rmin < cnst.LENGTH_TOLERANCE) # no inner spherical surface
        flag2 = (dphi > 360.-cnst.ANGLE_TOLERANCE) # no lateral planar faces
        flag3 = (stheta < cnst.ANGLE_TOLERANCE) # no upper cone face
        flag4 = (stheta+dtheta > 180.-cnst.ANGLE_TOLERANCE) # no lower cone face

        if (flag1 and flag2 and flag3 and flag4):
            return 0 # outer_sph
        elif (flag1 and not flag2 and flag3 and flag4):
            return 1 # outer_sph, planar_min_phi, planar_max_phi
        elif (not flag1 and flag2 and flag3 and flag4):
            return 2 # outer_sph, inner_sph
        elif (not flag1 and not flag2 and flag3 and flag4):
            return 3 # outer_sph, inner_sph, planar_min_phi, planar_max_phi
        elif (flag1 and flag2 and not flag3 and flag4):
            return 4 # outer_sph, conical_min_theta
        elif (flag1 and not flag2 and not flag3 and flag4):
            return 5 # outer_sph, planar_min_phi, planar_max_phi, conical_min_theta
        elif (not flag1 and flag2 and not flag3 and flag4):
            return 6 # outer_sph, inner_sph, conical_min_theta
        elif (not flag1 and not flag2 and not flag3 and flag4):
            return 7 # outer_sph, inner_sph, planar_min_phi, planar_max_phi, conical_min_theta

        elif (flag1 and flag2 and flag3 and not flag4):
            return 8 # outer_sph, conical_max_theta
        elif (flag1 and not flag2 and flag3 and not flag4):
            return 9 # outer_sph, planar_min_phi, planar_max_phi, conical_max_theta
        elif (not flag1 and flag2 and flag3 and not flag4):
            return 10 # outer_sph, inner_sph, conical_max_theta
        elif (not flag1 and not flag2 and flag3 and not flag4):
            return 11 # outer_sph, inner_sph, planar_min_phi, planar_max_phi, conical_max_theta
        elif (flag1 and flag2 and not flag3 and not flag4):
            return 12 # outer_sph, conical_min_theta, conical_max_theta
        elif (flag1 and not flag2 and not flag3 and not flag4):
            return 13 # outer_sph, planar_min_phi, planar_max_phi, conical_min_theta, conical_max_theta
        elif (not flag1 and flag2 and not flag3 and not flag4):
            return 14 # outer_sph, inner_sph, conical_min_theta, conical_max_theta
        elif (not flag1 and not flag2 and not flag3 and not flag4):
            return 15 # outer_sph, inner_sph, planar_min_phi, planar_max_phi, conical_min_theta, conical_max_theta

    def get_faces(self):
        assert(self._constructed==True)
        subtype = self.get_subtype()
        if (subtype == 0):
            return [0]
        elif (subtype == 1):
            return [0, 2, 3]
        elif (subtype == 2):
            return [0, 1]
        elif (subtype == 3):
            return [0, 1, 2, 3]
        elif (subtype == 4):
            return [0, 4]
        elif (subtype == 5):
            return [0, 2, 3, 4]
        elif (subtype == 6):
            return [0, 1, 4]
        elif (subtype == 7):
            return [0, 1, 2, 3, 4]

        elif (subtype == 8):
            return [0, 5]
        elif (subtype == 9):
            return [0, 2, 3, 5]
        elif (subtype == 10):
            return [0, 1, 5]
        elif (subtype == 11):
            return [0, 1, 2, 3, 5]
        elif (subtype == 12):
            return [0, 4, 5]
        elif (subtype == 13):
            return [0, 2, 3, 4, 5]
        elif (subtype == 14):
            return [0, 1, 4, 5]
        elif (subtype == 15):
            return [0, 1, 2, 3, 4, 5]

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
        stheta = rnd.uniform(0., 180.)
        dtheta = rnd.uniform(0., 180.-stheta)
        self._params = [rmin,
                        rmax,
                        rnd.uniform(0., 360.),
                        rnd.uniform(0., 360.),
                        stheta,
                        dtheta]
        self._constructed = True

    # Output point shape (3)
    # [xyz]
    # TODO implement boundary for the planar and conical faces
    def gen_point_inside(self):
        assert (self._constructed==True)
        rmin = self._params[0]
        rmax = self._params[1]
        sphi = self._params[2]
        dphi = self._params[3]
        stheta = self._params[4]
        dtheta = self._params[5]
        bth = cnst.BOUNDARY_THICKNESS/2 # boundary thickness half
        r = rnd.uniform(rmin+bth, rmax-bth)
        phi = rnd.uniform(sphi, sphi+dphi)
        theta = rnd.uniform(stheta, stheta+dtheta)
        x = r * np.cos(phi*cnst.DEGTORAD) * np.sin(theta*cnst.DEGTORAD)
        y = r * np.sin(phi*cnst.DEGTORAD) * np.sin(theta*cnst.DEGTORAD)
        z = r * np.cos(theta*cnst.DEGTORAD)
        return [x, y, z]

    # Output point shape (3)
    # [xyz]
    def gen_point_on_boundary(self):
        assert(self._constructed==True)
        rmin = self._params[0]
        rmax = self._params[1]
        sphi = self._params[2]
        dphi = self._params[3]
        stheta = self._params[4]
        dtheta = self._params[5]
        face_id = self.get_random_face_id()
        if (face_id == 0): # outer_sph
            r = rmax
            phi = rnd.uniform(sphi, sphi+dphi)
            theta = rnd.uniform(stheta, stheta+dtheta)
        elif (face_id == 1): # inner_sph
            r = rmin
            phi = rnd.uniform(sphi, sphi+dphi)
            theta = rnd.uniform(stheta, stheta+dtheta)
        elif (face_id == 2): # planar_min_phi
            r = rnd.uniform(rmin, rmax)
            phi = sphi
            theta = rnd.uniform(stheta, stheta+dtheta)
        elif (face_id == 3): # planar_max_phi
            r = rnd.uniform(rmin, rmax)
            phi = sphi+dphi
            theta = rnd.uniform(stheta, stheta+dtheta)
        elif (face_id == 4): # conical_min_theta
            r = rnd.uniform(rmin, rmax)
            phi = rnd.uniform(sphi, sphi+dphi)
            theta = stheta
        elif (face_id == 5): # conical_max_theta
            r = rnd.uniform(rmin, rmax)
            phi = rnd.uniform(sphi, sphi+dphi)
            theta = stheta+dtheta
        x = r * np.cos(phi*cnst.DEGTORAD) * np.sin(theta*cnst.DEGTORAD)
        y = r * np.sin(phi*cnst.DEGTORAD) * np.sin(theta*cnst.DEGTORAD)
        z = r * np.cos(theta*cnst.DEGTORAD)
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

    shp = csg_sphere()
    shp.gen_random()
    cloud = shp.gen_cloud_on_boundary(10000)

    draw_cloud(1, cloud, "SPHERE")
    plt.show()

    exit()
