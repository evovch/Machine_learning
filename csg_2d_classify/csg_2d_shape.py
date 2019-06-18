import numpy as np
import random as rnd

# ==============================================================================

class csg_2d_shape():

    def __init__(self):
        self._primitive_type = -1 #TODO check
        self._param_names = []
        self._params = []
        self._face_names = []
        self._constructed = False

    def get_primitive_type(self):
        assert(self._constructed==True)
        return self._primitive_type

    def get_n_faces(self):
        assert(self._constructed==True)
        faces = self.get_faces()
        return len(faces)

    def get_random_face_id(self):
        assert(self._constructed==True)
        faces = self.get_faces()
        idx = rnd.choice(range(self.get_n_faces()))
        return faces[idx]

    # # Output cloud shape (n_points,2):
    # # [[xy][xy]...[xy]]
    # def gen_cloud_inside(self, n_points):
    #     assert(self._constructed==True)
    #     cloud = np.zeros((n_points,2))
    #     for i in range(n_points):
    #         cloud[i] = self.gen_point_inside()
    #     return cloud

    # Output cloud shape (n_points,2):
    # [[xy][xy]...[xy]]
    def gen_cloud_on_boundary(self, n_points):
        assert(self._constructed==True)
        cloud = np.zeros((n_points,2))
        for i in range(n_points):
            cloud[i] = self.gen_point_on_boundary()
        return cloud

# ==============================================================================

def get_primitive_name(primitive_type):
    names = ['square','rectangle','trapezoid','generic_trapezoid','para','full_circle','full_ellipse','cirlce','ellipse']
    if (primitive_type>=0 and primitive_type<len(names)):
        return names[primitive_type]
    else:
        return names[len(names)-1]

def get_primitive_names(primitive_types):
    names = []
    for i in range(len(primitive_types)):
        names.append(get_primitive_name(primitive_types[i]))
    return names

# ==============================================================================