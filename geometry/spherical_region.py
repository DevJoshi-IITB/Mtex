import numpy as np
from vector3d import Vector3d
from rotation import Rotation

class SphericalRegion:
    def __init__(self, N=Vector3d(), alpha=[], antipodal=False, vertices:Vector3d=None, max_theta=np.pi, min_theta=0, max_rho=2*np.pi, min_rho=0):
        self.N = N
        if alpha:
            self.alpha = np.array(alpha)
            if np.isscalar(self.alpha):
                self.alpha = np.full(len(self.N), self.alpha)
        else:
            self.alpha = np.zeros(len(self.x))
        self.antipodal = antipodal
        
        self.vertices = vertices
        if vertices:
            shifted_vertices = vertices[2:].extend(vertices[1])
            N = Vector3d.cross(vertices, shifted_vertices).normalize()
            self.N = self.N.extend(N)
            self.alpha = np.concatenate((self.alpha, np.zeros(len(self.N))))

        #additional options
        if max_theta < np.pi - 1e-5:
            self.N = self.N.extend(Vector3d(0,0,1))
            self.alpha = np.concatenate((self.alpha, np.cos(max_theta)))
        if min_theta > 1e-5:
            self.N = self.N.extend(Vector3d(0,0,-1))
            self.alpha = np.concatenate((self.alpha, np.cos(np.pi - min_theta)))

        if max_rho - min_rho < 2*np.pi - 1e-5:
            self.N = self.N.extend(Vector3d.by_polar([np.pi/2,np.pi/2],[np.pi/2 + min_rho,max_rho - np.pi/2]))
            self.alpha = np.concatenate((self.alpha, np.array([0,0])))

    @property
    def how2plot(self):
        return self.N.plotting_convention
    
    def copy(self):
        return SphericalRegion(self.N, self.alpha, self.antipodal)

    def theta_range(self, rho=0):
        if not self.N or self.N.isempty() or self.N.isnan() or self.N.size == 0:
            theta_min = np.zeros_like(rho)
            theta_max = np.pi * np.ones_like(rho)
            return theta_min, theta_max
        elif len(self.N) == 1 and self.N == Vector3d.Z and np.all(self.alpha == 0):
            theta_min = np.zeros_like(rho)
            theta_max = np.pi/2 * np.ones_like(rho)
            return theta_min, theta_max
        elif len(self.N) == 1 and self.N == -Vector3d.Z and np.all(self.alpha == 0):
            theta_min = np.pi/2 * np.zeros_like(rho)
            theta_max = np.pi * np.ones_like(rho)
            return theta_min, theta_max
    
        self.antipodal = False

        if isinstance(rho, np.ndarray):
            theta = np.linspace(0, np.pi, 10001)
            rho_grid, theta_grid = np.meshgrid(rho, theta)
            v = Vector3d.by_polar(theta_grid.flatten(), rho_grid.flatten())
            inside = self.check_inside(v)
            theta_grid[~inside.reshape(theta_grid.shape)] = np.nan
            theta_min = np.nanmin(theta_grid, axis=0)
            theta_max = np.nanmax(theta_grid, axis=0)
            return theta_min, theta_max
        else:
            th = self.vertices.theta  if self.vertices else None

            if self.check_inside(Vector3d.Z):
                theta_min = 0
            elif th and th.size != 0:
                theta_min = np.min(th)
            elif self.N and len(self.N) > 0:
                theta_min = np.pi - max(np.arccos(self.alpha) + Vector3d.angle(self.N, -Vector3d.Z)) 
            else:
                theta_min = 0

            if self.check_inside(-Vector3d.Z):
                theta_max = np.pi
            elif th and th.size != 0:
                theta_max = np.max(th)
            elif self.N and len(self.N) > 0:
                theta_max = max(np.arccos(self.alpha) + Vector3d.angle(self.N, Vector3d.Z))
            else:
                theta_max = np.pi

            return theta_min, theta_max
        
    def theta_min(self):
        return self.theta_range()[0]
    
    def theta_max(self):
        return self.theta_range()[1]

    #TODO implement this method    
    def rho_range(self):
        if self.N.x.size == 0 or (self.N.z != 0 and np.all(self.alpha == 0)):
            return 0, 2*np.pi
        
        self.antipodal = False

        omega = np.linspace(0, 2*np.pi, 361)

        v = Vector3d.by_polar(np.full(omega.shape, np.pi/2), omega)
        rho = v.rho[self.check_inside(v)]

        for i in range(len(self.N)):
            b = Vector3d.by_polar(np.arccos(self.alpha[i]),omega)
            rot = Rotation.map(Vector3d.Z, self.N[i])



    def rho_min(self):
        return self.rho_range()[0]
    
    def rho_max(self):
        return self.rho_range()[1]
                
    def check_inside(self, v:Vector3d, no_antipodal=False):
        sr = self.copy()
        if  sr.antipodal and not no_antipodal:
            sr.antipodal = False
            return sr.check_inside(v) or sr.check_inside(-v)
        
        if no_antipodal:
            v.antipodal = False

        inside = self.N*v >= (self.alpha - 1e-4)
        return np.all(inside,axis=-1)



            
