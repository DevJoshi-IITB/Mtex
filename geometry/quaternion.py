import numpy as np
from vector3d import Vector3d

class Quaternion:
    def __init__(self, a=0, b=0, c=0, d=0):
        self.a = np.array(a, dtype=float, ndmin=1)      #real part
        self.b = np.array(b, dtype=float, ndmin=1)      #i
        self.c = np.array(c, dtype=float, ndmin=1)      #j
        self.d = np.array(d, dtype=float, ndmin=1)      #k

    def __eq__(self, other):
        if isinstance(other, Quaternion):
            return np.all(self.a == other.a) and np.all(self.b == other.b) and np.all(self.c == other.c) and np.all(self.d == other.d)
        return False

    def __add__(self,other):
        if isinstance(other,Quaternion):
            return Quaternion(self.a + other.a, self.b + other.b, self.c + other.c, self.d + other.d)
        elif np.isscalar(other):
            return Quaternion(self.a + other, self.b + other, self.c + other, self.d + other)
        raise TypeError(f"Addition of vectors is not possible with {type(other)}")
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(self.a - other.a, self.b - other.b, self.c - other.c, self.d - other.d)
        elif np.isscalar(other):
            return Quaternion(self.a - other, self.b - other, self.c - other, self.d - other)
        raise TypeError(f"Subtraction of vectors is not possible with {type(other)}")
    
    def __rsub__(self, other):
        return self.__sub__(other)
    
    def __mul__(self, other):
        if isinstance(other, Quaternion):
            return self.a * other.a + self.b * other.b + self.c * other.c + self.d * other.d
        elif np.isscalar(other):
            return Quaternion(self.a * other, self.b * other, self.c * other, self.d * other)
        raise TypeError(f"Multiplication of vectors is not possible with {type(other)}")
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if np.isscalar(other):
            return Quaternion(self.a / other, self.b / other, self.c / other, self.d / other)
        raise TypeError(f"Division of vectors is not possible with {type(other)}")
    
    def __neg__(self):
        return Quaternion(-self.a, -self.b, -self.c, -self.d)
    
    def __str__(self):
        res = ""
        for i in range(len(self)):
            res = res + str(self.a[i]) + " " + str(self.b[i]) + " " + str(self.c[i]) + " " + str(self.d[i]) + '\n'

        return res
    
    def __repr__(self):
        res = ""
        for i in range(len(self)):
            res = res + str(self.a[i]) + " " + str(self.b[i]) + " " + str(self.c[i]) + " " + str(self.d[i]) + '\n'

        return res
    
    def __getitem__(self, i):
        return Quaternion(self.a[i], self.b[i], self.c[i], self.d[i])
    
    def __setitem__(self, i, value):
        if isinstance(value, (tuple,list)):
            self.a[i] = value[0]
            self.b[i] = value[1]
            self.c[i] = value[2]
            self.d[i] = value[3]
        elif isinstance(value, Quaternion):
            self.a[i] = value.a
            self.b[i] = value.b
            self.c[i] = value.c
            self.d[i] = value.d
        else:
            raise TypeError(f"Cannot set value of type {type(value)}")

    def __len__(self):
        return len(self.a)

    def __iter__(self):
        self.start = 0
        return self

    def __next__(self):
        if self.start >= len(self):
            raise StopIteration
        i = self.start
        self.start += 1
        return Quaternion(self.a[i], self.b[i], self.c[i], self.d[i])    

    @classmethod
    def from_vectors(cls, vec:Vector3d):
        return cls(np.zeros(vec.x.shape), vec.x, vec.y, vec.z)
    
    def abcd(self):
        return np.column_stack((self.a, self.b, self.c, self.d))
    
    def norm(self):
        return np.sqrt(self.a**2 + self.b**2 + self.c**2 + self.d**2)
    
    def normalize(self):
        norm = self.norm()
        if np.all(norm > 0):
            self.a /= norm
            self.b /= norm
            self.c /= norm
            self.d /= norm
        return self
    
    @staticmethod
    def quat_dot(q1:'Quaternion', q2):
        return q1.a*q2.a + q1.b*q2.b + q1.c*q2.c + q1.d*q2.d
    
    @staticmethod
    def quat_dot_outer(q1:'Quaternion', q2:'Quaternion'):
        return np.dot(q1.abcd(), q2.abcd().T) if q1.abcd().size != 0 and q2.abcd().size != 0 else []
    
    @staticmethod
    def nan(shape):
        return Quaternion(np.full(shape, np.nan), np.full(shape, np.nan), np.full(shape, np.nan), np.full(shape, np.nan))
    
    @staticmethod
    def id(shape):
        return Quaternion(np.ones(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape))
    
    #TODO implement this method and recreate all the required files
    @staticmethod
    def rand(shape):
        pass
    
    @staticmethod
    def axis2quat(v: Vector3d, omega):
        norm = v.norm()
        a = np.cos(omega/2)
        b = np.sin(omega/2) * v.x / norm
        c = np.sin(omega/2) * v.y / norm
        d = np.sin(omega/2) * v.z / norm
        return Quaternion(a, b, c, d)
    
    @staticmethod
    def euler2quat(alpha, beta, gamma):
        alpha = np.array(alpha, dtype=float)
        beta = np.array(beta, dtype=float)
        gamma = np.array(gamma, dtype=float)
        zero = np.zeros_like(alpha)
        qalpha = Quaternion(np.cos(alpha/2), zero, zero, np.sin(alpha/2))
        qbeta = Quaternion(np.cos(beta/2), zero, np.sin(beta/2), zero)
        qgamma = Quaternion(np.cos(gamma/2), zero, zero, np.sin(gamma/2))
        return qalpha * qbeta * qgamma

    @staticmethod
    def mat2quat(mat):
        n = mat.shape[2]
        Quat = np.zeros((n, 4))
    
        absQ = np.zeros((4, n))
        absQ[0, :] = 0.5 * np.sqrt(1 + mat[0, 0, :] + mat[1, 1, :] + mat[2, 2, :])
        absQ[1, :] = 0.5 * np.sqrt(1 - mat[0, 0, :] - mat[1, 1, :] + mat[2, 2, :])
        absQ[2, :] = 0.5 * np.sqrt(1 + mat[0, 0, :] - mat[1, 1, :] - mat[2, 2, :])
        absQ[3, :] = 0.5 * np.sqrt(1 - mat[0, 0, :] + mat[1, 1, :] - mat[2, 2, :])

        ind = np.argmax(absQ, axis=0)

        for i in range(n):
            if ind[i] == 0:
                Quat[i, 0] = absQ[0, i]
                Quat[i, 1] = (mat[1, 2, i] - mat[2, 1, i]) * 0.25 / absQ[0, i]
                Quat[i, 2] = (mat[2, 0, i] - mat[0, 2, i]) * 0.25 / absQ[0, i]
                Quat[i, 3] = (mat[0, 1, i] - mat[1, 0, i]) * 0.25 / absQ[0, i]
            elif ind[i] == 1:
                Quat[i, 0] = (mat[0, 1, i] - mat[1, 0, i]) * 0.25 / absQ[1, i]
                Quat[i, 1] = absQ[1, i]
                Quat[i, 2] = (mat[2, 0, i] + mat[0, 2, i]) * 0.25 / absQ[1, i]
                Quat[i, 3] = (mat[2, 1, i] + mat[1, 2, i]) * 0.25 / absQ[1, i]
            elif ind[i] == 2:
                Quat[i, 0] = (mat[1, 2, i] - mat[2, 1, i]) * 0.25 / absQ[2, i]
                Quat[i, 1] = (mat[0, 1, i] + mat[1, 0, i]) * 0.25 / absQ[2, i]
                Quat[i, 2] = absQ[2, i]
                Quat[i, 3] = (mat[2, 0, i] + mat[0, 2, i]) * 0.25 / absQ[2, i]
            elif ind[i] == 3:
                Quat[i, 0] = (mat[2, 0, i] - mat[0, 2, i]) * 0.25 / absQ[3, i]
                Quat[i, 1] = (mat[0, 1, i] + mat[1, 0, i]) * 0.25 / absQ[3, i]
                Quat[i, 2] = (mat[1, 2, i] + mat[2, 1, i]) * 0.25 / absQ[3, i]
                Quat[i, 3] = absQ[3, i]

        Quat = np.real(Quat)
        return Quaternion(Quat[0], -Quat[1], -Quat[2], -Quat[3]).normalize()

    @staticmethod
    def hr2quat(h:Vector3d, r:Vector3d):
        h = h.normalize()
        r = r.normalize()
        h.antipodal = False
        r.antipodal = False
        n = Vector3d.cross(h, r)
        
        if np.allclose(n,0):
            n = h.orth() if len(h) >= len(r) else r.orth()

        return Quaternion.axis2quat(n, Vector3d.angle(h,r))



