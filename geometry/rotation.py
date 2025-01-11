import numpy as np
from quaternion import Quaternion
from vector3d import Vector3d
from copy import deepcopy

#TODO dunder methods for arithmatic, etc...
class Rotation(Quaternion):
    def __init__(self,q:Quaternion=None, i=False, method=None, axis=None, angle=None, angles=None, matrix=None, normal:Vector3d=None, vectors:Vector3d=None, vector:Vector3d=None):
        super().__init__(q.a, q.b, q.c, q.d)
        self.i = np.array(i, dtype=np.bool, ndmin=1)

        if method is not None:
            if method == 'axis':
                loc_rot = self.by_axis_angle((axis, angle))
            elif method == 'euler':
                loc_rot = self.by_euler(angles[0], angles[1], angles[2])
            elif method == 'map':
                loc_rot = self.map(vectors[0], vectors[1], vectors[2], vectors[3]) if len(vectors) == 4 else self.map(vectors[0], vectors[1])
            elif method == 'rodrigues':
                loc_rot = self.by_rodrigues(vector)
            elif method == 'matrix':
                loc_rot = self.by_matrix(matrix)
            elif method in ('mirroring', 'reflection'):
                loc_rot = self.reflection(normal)

            else:
                raise ValueError(f"Unknown rotation method: {method}")

            self.a, self.b, self.c, self.d = loc_rot.a, loc_rot.b, loc_rot.c, loc_rot.d
            self.i = loc_rot.i
                
    def __mul__(self,other):
        if isinstance(other, list):
            other = np.array(other)
        if isinstance(other,np.ndarray) and np.issubdtype(other.dtype, np.number):
            if len(self.a) == other.shape[1]:
                r = deepcopy(self)
                r.a = other @ r.a
                r.b = other @ r.b
                r.c = other @ r.c
                r.d = other @ r.d
                r.i = other @ r.i
                r.normalize()
                r.i = r.i > 0.5
            else:
                assert np.all(np.abs(other) == 1),"Rotations can be multiplied only by 1 or -1"
                tmp = Rotation.id(np.size(other))
                tmp.i = (1 - other) / 2
                r = deepcopy(tmp)
            return r
        elif isinstance(other, Vector3d):
            return Vector3d.rotate_outer(other, self)
        elif isinstance(other, Quaternion):
            return other * self
        raise TypeError(f"Multiplication of rotation with {type(other)} is not possible")    


    @staticmethod
    def by_axis_angle(v:Vector3d, omega):
        return Rotation(Quaternion.axis2quat(v, omega))
    
    @staticmethod
    def by_euler(alpha, beta, gamma):
        return Rotation(Quaternion.euler2quat(alpha, beta, gamma))
    
    @staticmethod
    def map(u1:Vector3d, v1:Vector3d, u2:Vector3d = None, v2:Vector3d = None):
        if u2 and v2:
            u1 = u1.normalize()
            v1 = v1.normalize()
            u2 = u2.normalize()
            v2 = v2.normalize()

            delta = np.abs(Vector3d.angle(u1, u2) - Vector3d.angle(u2, v2))

            assert np.all(delta < 1e-3), f"Inconsitent pairs of vectors! The angle between u1, u2 and v1, v2 needs to be the same, but differs by {np.degrees(np.max(delta))}°"

            if np.any(np.abs(u1*u2) > 1 - np.finfo(np.finfo(float).eps)):
                raise ValueError("Input vectors should not be colinear!")
            
            u3 = Vector3d.cross(u1, u2).normalize()
            v3 = Vector3d.cross(v1, v2).normalize()

            u2t = Vector3d.cross(u3, u1).normalize()
            v2t = Vector3d.cross(v3, v1).normalize()

            A = np.stack([v1.xyz(), v2t.xyz(), v3.xyz()], axis=1)
            B = np.stack([u1.xyz(), u2t.xyz(), u3.xyz()], axis=1)
            M = A @ np.linalg.inv(B)

            return Rotation.by_matrix(M)
        else:
            return Rotation(Quaternion.hr2quat(u1,v1))

    @staticmethod
    def by_matrix(M):
        isinv = np.zeros(M.shape[2], dtype=bool)
        for i in range(M.shape[2]):
            isinv[i] = np.linalg.det(M[:,:,i]) < 0
        M[:,:,i] = -M[:,:,i]
        return Rotation(Quaternion.mat2quat(M), isinv)
    
    @staticmethod
    def by_rodrigues(v:Vector3d, theta):
        return Rotation.by_axis_angle(v, 2*np.arctan(v.norm()))
    
    @staticmethod
    def reflection(n:Vector3d):
        return -Rotation.by_axis_angle(n, np.pi)

    @staticmethod
    def nan(shape):
        return Rotation(Quaternion.nan(shape))
    
    @staticmethod
    def id(shape):
        return Rotation(Quaternion.id(shape))
    
    #TODO implement quaterion rand, vector rand for rotation rand
    @staticmethod
    def rand(shape):
        pass

    @staticmethod
    def inversion(shape):
        return Rotation(-Quaternion.inversion(shape))
    
    #TODO vector3d.eig>mathtools.eig3, rotation.rand,s3g.localorientatoingrid
    @staticmethod
    def fit(args):
        pass

    @staticmethod
    def load(args):
        pass

