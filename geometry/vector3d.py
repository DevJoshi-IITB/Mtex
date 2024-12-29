import os
import numpy as np

#FIXME error handling
class Vector3d:
    def __init__(self, x=[], y=[], z=[], antipodal=False, isNormalized=False, normalize=False, plotting_convention=None): 
        self.x = np.array(x , dtype=float, ndmin=1)
        self.y = np.array(y , dtype=float, ndmin=1)
        self.z = np.array(z , dtype=float, ndmin=1)
        self.antipodal = antipodal
        self.isNormalized = isNormalized
        self.plotting_convention = plotting_convention
        if normalize:
            self.normalize()

    def __eq__(self, other):
        if isinstance(other, Vector3d):
            return np.all(self.x == other.x) and np.all(self.y == other.y) and np.all(self.z == other.z)
        return False

    def __add__(self,other):
        if isinstance(other,Vector3d):
            return Vector3d(self.x + other.x, self.y + other.y, self.z + other.z)
        elif np.isscalar(other):
            return Vector3d(self.x + other, self.y + other, self.z + other)
        raise TypeError(f"Addition of vectors is not possible with {type(other)}")
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, Vector3d):
            return Vector3d(self.x - other.x, self.y - other.y, self.z - other.z)
        elif np.isscalar(other):
            return Vector3d(self.x - other, self.y - other, self.z - other)
        raise TypeError(f"Subtraction of vectors is not possible with {type(other)}")
    
    def __rsub__(self, other):
        return self.__sub__(other)
    
    def __mul__(self, other):
        if isinstance(other, Vector3d):
            dp = self.x * other.x + self.y * other.y + self.z * other.z
            return abs(dp) if self.antipodal or other.antipodal else dp
        elif np.isscalar(other):
            return Vector3d(self.x * other, self.y * other, self.z * other)
        raise TypeError(f"Multiplication of vectors is not possible with {type(other)}")
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if np.isscalar(other):
            return Vector3d(self.x / other, self.y / other, self.z / other)
        raise TypeError(f"Division of vectors is not possible with {type(other)}")
    
    def __neg__(self):
        return Vector3d(-self.x, -self.y, -self.z)
    
    def __str__(self):
        res = ""
        for i in range(len(self)):
            res = res + str(self.x[i]) + " " + str(self.y[i]) + " " + str(self.z[i]) + '\n'

        return res
    
    def __repr__(self):
        res = ""
        for i in range(len(self)):
            res = res + str(self.x[i]) + " " + str(self.y[i]) + " " + str(self.z[i]) + '\n'

        return res
    
    
    def __getitem__(self, i):
        return Vector3d(self.x[i], self.y[i], self.z[i])
    
    def __setitem__(self, i, value):
        if isinstance(value, (tuple, list)):
            self.x[i] = value[0]
            self.y[i] = value[1]
            self.z[i] = value[2]
        elif isinstance(value, Vector3d):
            self.x[i] = value.x
            self.y[i] = value.y
            self.z[i] = value.z
        else:
            raise TypeError(f"Cannot set value of type {type(value)}")

    def __len__(self):
        return len(self.x)

    def __iter__(self):
        self.start = 0
        return self

    def __next__(self):
        if self.start >= len(self):
            raise StopIteration
        i = self.start
        self.start += 1
        return Vector3d(self.x[i], self.y[i], self.z[i])    

    @property
    def X(self):
        return Vector3d(1,0,0)
    
    @property
    def Y(self):
        return Vector3d(0,1,0)
    
    @property
    def Z(self):
        return Vector3d(0,0,1)

    @property
    def rho(self):
        rho = np.arctan2(self.y, self.x)
        return rho + (rho < 0) * 2 * np.pi

    
    @property
    def theta(self):
        norm = self.norm()
        theta =  np.arccos(np.divide(self.z, norm, where= norm != 0))
        theta[norm == 0] = 0
        return theta

    @property
    def resolution(self):
        length = len(self.x)
        if length <= 4:
            return 2 * np.pi
        elif length > 50000:
            return np.sqrt(40000 / length / (1 + int(self.antipodal))) * (np.pi / 180)  
        
        try:
            area,_ = Vector3d.calc_voronoi_area(self)
            res = np.sqrt(np.median(area))
            assert res > 0, "Resolution must be positive"

        except Exception as e:
            print(f"Error calculating resolution: {e}")
            res = 2 * np.pi
        
        return res

    def as_polar(self, degrees=False):
        res = ""
        theta = self.theta if not degrees else np.degrees(self.theta)
        rho = self.rho if not degrees else np.degrees(self.rho)
        for i in range(len(self)):
            res = res + str(theta[i]) + " " + str(rho[i]) + '\n'

        return res
        
    def sub_set(self,indices):
        return Vector3d(self.x[indices], self.y[indices], self.z[indices])
    
    def extend(self,v:'Vector3d'):
        self.x = np.concatenate((self.x, v.x))
        self.y = np.concatenate((self.y, v.y))
        self.z = np.concatenate((self.z, v.z))
        return self
        
    def norm(self):
        return np.array(np.sqrt(self.x**2 + self.y**2 + self.z**2))
    
    def xyz(self):
        return np.column_stack((self.x, self.y, self.z))
    
    def xy(self):
        return np.column_stack((self.x, self.y))
    
    def normalize(self, v:'Vector3d'):
        if v:
            norm = v.norm()
            if np.all(norm > 0):
                v.x /= norm
                v.y /= norm
                v.z /= norm
                v.isNormalized = True
            return v
        norm = self.norm()
        if np.all(norm > 0):
            self.x /= norm
            self.y /= norm
            self.z /= norm
            self.isNormalized = True
        return self
        

    def reshape(self, shape):
        return Vector3d(self.x.reshape(shape), self.y.reshape(shape), self.z.reshape(shape))
    
    def isinf(self):
        return np.isinf(self.x) or np.isinf(self.y) or np.isinf(self.z)
    
    def isfinite(self):
        return not (self.isinf() or self.isnan())
    
    def isreal(self):
        return np.isreal(self.x) and np.isreal(self.y) and np.isreal(self.z)

    def isnan(self):
        return np.isnan(self.x) | np.isnan(self.y) | np.isnan(self.z)
    
    def isempty(self):
        return len(self.x) == 0 or len(self.y) == 0 or len(self.z) == 0
    
    def real(self):
        return Vector3d(np.real(self.x), np.real(self.y), np.real(self.z))
    
    def imag(self):
        return Vector3d(np.imag(self.x), np.imag(self.y), np.imag(self.z))

    def save(self, fname, delimiter=',', as_polar=False):
        with open(fname, 'w') as file:
            if as_polar:
                file.write(str(self.as_polar()).replace(' ',delimiter))
            else:
                file.write(str(self).replace(' ', delimiter))

    
    def calc_voronoi_area(vec:'Vector3d'):
        from scipy.spatial import Voronoi
        v = vec.normalize()
        v = v.xyz()
        if v.antipodal:
            v = np.vstack((v, -v))

        vor = Voronoi(v)
    
        areas = np.zeros(len(v))
        centroids = np.zeros_like(v)

        for i, region in enumerate(vor.regions):
            if not region or -1 in region:
                continue

            vertices = vor.vertices[region] / np.linalg.norm(vertices, axis=1, keepdims=True)

            tri_areas = []
            tri_centroids = []
            for j in range(len(vertices)):
                va = v[i]
                vb = vertices[j]
                vc = vertices[(j + 1) % len(vertices)]

                tri_area = Vector3d.spherical_triangle_area(va,vb,vc)
                tri_areas.append(tri_area)

                tri_centroid = (va + vb + vc) / 3
                tri_centroids.append(tri_centroid)

            areas[i] = np.sum(tri_areas)
            centroids[i] = np.sum(tri_centroids, axis=0)

        centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)

        if v.antipodal:
            unique_areas = areas[:len(areas)//2]
            unique_areas /= 2
            areas = unique_areas
            centroids = centroids[:len(centroids)//2]

        return areas, centroids

    def spherical_triangle_area(va, vb, vc):
        angle_a = np.arccos(np.clip(np.dot(vb, vc), -1, 1))
        angle_b = np.arccos(np.clip(np.dot(vc, va), -1, 1))
        angle_c = np.arccos(np.clip(np.dot(va, vb), -1, 1))

        spherical_excess = angle_a + angle_b + angle_c - np.pi
        return spherical_excess
    
    def sum(self):
        return Vector3d(np.sum(self.xyz(), axiz=0))

    def orth(self):
        if not self.x:
            return Vector3d.X(len(self.x))
        else:
            return Vector3d(-self.y, self.x, np.zeros_like(self.x)).normalize()
    
    @staticmethod
    def by_polar(theta, rho, degrees=False):
        if degrees:
            theta = np.radians(theta)
            rho = np.radians(rho)
        x = np.sin(theta) * np.cos(rho)
        y = np.sin(theta) * np.sin(rho)
        z = np.cos(theta)
        return Vector3d(x, y, z, isNormalized=True)
    
    @staticmethod
    def zeros(shape):
        return Vector3d(np.zeros(shape), np.zeros(shape), np.zeros(shape))

    @staticmethod
    def ones(shape):
        return Vector3d(np.ones(shape), np.ones(shape), np.ones(shape))
    
    @staticmethod
    def nan(shape):
        return Vector3d(np.full(shape, np.nan), np.full(shape, np.nan), np.full(shape, np.nan))
    
    @staticmethod
    def load(fname, column_names=None, columns=None, degrees=False, delimiter=',', header=0):
        if not os.path.isfile(fname):
            raise FileNotFoundError(f"File {fname} not found.")
    
        if fname.endswith(".csv"):
            delimiter = ','
        else:
            delimiter = delimiter or ' '

        try:
            data = np.loadtxt(fname, delimiter=delimiter)
            if column_names:
                column_headers = data[0]
                columns = np.where(np.isin(column_headers, column_names))
                header = 1
            if not columns:
                columns = np.arange(data.shape[1])
            data = data[header:,columns]

            if data.shape[1] == 3:
                return Vector3d(data[:, 0], data[:, 1], data[:, 2])
            elif data.shape[1] == 2:
                return Vector3d.byPolar(data[:, 0], data[:, 1], degrees=degrees)
            
        except Exception as e:
            raise RuntimeError(f"An error occured while reading the file: {e}")
 
    #TODO make sphericalRegion class 
    @staticmethod
    def rand():
        
        pass
    
    @staticmethod
    def X(shape=1):
        return Vector3d(np.ones(shape), np.zeros(shape), np.zeros(shape))
    
    @staticmethod
    def Y(shape=1):
        return Vector3d(np.zeros(shape), np.ones(shape), np.zeros(shape))
    
    @staticmethod
    def Z(shape=1):
        return Vector3d(np.zeros(shape), np.zeros(shape), np.ones(shape))
    
    @staticmethod
    def dot(v1:'Vector3d',v2:'Vector3d', antipodal=None):
        dp = v1.x*v2.x + v1.y*v2.y + v1.z*v2.z
        if antipodal is None:
            return abs(dp) if v1.antipodal or v2.antipodal else dp
        return abs(dp) if antipodal else dp 

    @staticmethod
    def cross(v1:'Vector3d', v2:'Vector3d'):
        x = v1.y * v2.z - v1.z * v2.y
        y = v1.z * v2.x - v1.x * v2.z
        z = v1.x * v2.y - v1.y * v2.x
        return Vector3d(x, y, z, antipodal= v1.antipodal or v2.antipodal)

    @staticmethod
    def angle(v1:'Vector3d', v2:'Vector3d', N=None):
        if N:
            N = Vector3d.normalize(N)
            v1 = Vector3d.normalize(v1 - Vector3d.dot(v1,N,antipodal=False) * N)
            v2 = Vector3d.normalize(v2 - Vector3d.dot(v2,N,antipodal=False) * N)
            
            theta = Vector3d.angle(v1, v2)

            if Vector3d.dot(N, Vector3d.cross(v1, v2), antipodal=False) < 0:
                theta = 2*np.pi - theta

            if v1.antipodal or v2.antipodal:
                theta = np.mod(theta, np.pi)

        else:
            theta = (v1 * v2) / (Vector3d.norm(v1) * Vector3d.norm(v2))
            return np.real(np.acos(theta))

    @staticmethod
    def mean(v:'Vector3d', robust=False, antipodal=False, weights=None):
        if robust and len(v.x) > 4:
            omega = v.angle(v)
            id = omega < np.quantile(omega, 0.8) * (1 + 1e-5)

            if np.any(id):
                return Vector3d.mean(v.sub_set(id))
            
        if antipodal:
            if weights:
                v.x *= np.sqrt(weights)
                v.y *= np.sqrt(weights)
                v.z *= np.sqrt(weights)

            xx = np.mean(v.x**2)
            xy = np.mean(v.x * v.y)
            xz = np.mean(v.x * v.z)
            yy = np.mean(v.y**2)
            yz = np.mean(v.y * v.z)
            zz = np.mean(v.z**2)

            cov_matrix = np.array([[xx,xy,xz],
                                   [xy,yy,yz],
                                   [xz,yz,zz]])
            eigvals, eigvals = np.linalg.eig(cov_matrix)
            m = Vector3d(*eigvals[:,np.argmax(eigvals)])

        else:
            if weights:
                v.x *= np.sqrt(weights)
                v.y *= np.sqrt(weights)
                v.z *= np.sqrt(weights)

            m = Vector3d(np.mean(v.x), np.mean(v.y), np.mean(v.z))
            m.isNormalized = False

        return m    
    
    @staticmethod
    def polar(v:'Vector3d'):
        r = v.norm()
        theta = v.theta
        rho = v.rho
        return theta, rho, r

