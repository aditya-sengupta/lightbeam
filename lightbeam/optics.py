import numpy as np
from matplotlib import pyplot as plt

from bisect import bisect_left, bisect_right
from functools import partial
from itertools import repeat
from numba import jit
from numpy import logical_and as AND, logical_not as NOT
from typing import List

from .geom import AA_circle_nonu
from .mesh import RectMesh2D
from .LPmodes import get_num_modes

### to do

# some ideas
#       -currently have outer bbox to speed up raster. maybe an inner bbox will also speed things up? this would force fancy indexing though...
#           -but the boundary region between inner and outer bbox could also be split into four contiguous blocks
#       -extension to primitives with elliptical cross sections
#           -I don't think this is that hard. An ellipse is a stretched circle. So antialiasing the ellipse on a rectangular grid is the same
#            as antialiasing a circle on another differently stretched rectangular grid.
#           -but as of now, this is unnecessary

class OpticPrim:
    '''base class for optical primitives (simple 3D shapes with a single IOR value)'''
    
    z_invariant = False
    
    def __init__(self, n):

        self.n = n
        self.n2 = n*n
        
        self.mask_saved = None

        # optionally, give prims a mesh to set the samplinig for IOR computations
        self.xymesh = None
    
    def _bbox(self,z):
        '''calculate the 2D bounding box of the primitive at given z. allows for faster IOR computation. Should be overwritten.'''
        return (-np.inf,np.inf,-np.inf,np.inf)

    def _contains(self,x,y,z):
        '''given coords, return whether or not those coords are inside the element. Should be overwritten.'''
        return np.full_like(x,False)

    def bbox_idx(self,z):
        '''get index slice corresponding to the primitives bbox, given an xg,yg coord grid'''
        m = self.xymesh
        xa, ya = m.xa, m.ya

        xmin,xmax,ymin,ymax = self._bbox(z)
        imin = max(bisect_left(xa,xmin) - 1, 0)
        imax = min(bisect_left(xa,xmax) + 1, len(xa))
        jmin = max(bisect_left(ya,ymin) - 1, 0)
        jmax = min(bisect_left(ya,ymax) + 1, len(ya))
        return np.s_[imin:imax, jmin:jmax], np.s_[imin:min(len(xa),imax+1),jmin:min(len(ya),jmax+1)]
    
    def set_sampling(self, xymesh:RectMesh2D):
        self.xymesh = xymesh
        self.rxg, self.ryg = np.meshgrid(xymesh.rxa, xymesh.rya, indexing='ij')
        self.dxg, self.dyg = np.meshgrid(xymesh.dxa, xymesh.dya, indexing='ij')

    def set_IORsq(self,out,z,coeff=1):
        ''' replace values of out with IOR^2, given coordinate grids xg, yg, and z location. 
            assumed primitive already has a set sampling grid'''

        if self.z_invariant and self.mask_saved is not None:
            mask = self.mask_saved
        else:
            bbox,bboxh = self.bbox_idx(z)
            mask = self._contains(self.xymesh.xg[bbox],self.xymesh.yg[bbox],z)

        if self.z_invariant and self.mask_saved is None:
            self.mask_saved = mask
        
        out[bbox][mask] = self.n2*coeff
    
    def get_boundary(self,z):
        ''' given z, get mask which will select pixels that lie on top of the primitive boundary
            you must set the sampling first before you call this!'''
        
        xhg, yhg = self.xymesh.xhg, self.xymesh.yhg
        maskh = self._contains(xhg,yhg,z)
        return NOT(
            AND(
                AND(
                    maskh[1:,1:] == maskh[:-1,1:],
                    maskh[1:,1:] == maskh[1:,:-1]
                ),
                maskh[:-1,:-1]==maskh[:-1,1:]
            )
        )

class ScaledCyl(OpticPrim):
    ''' cylinder whose offset from origin and radius scale in the same way'''
    def __init__(self,xy,r,z_ex,n,nb,z_offset=0,scale_func=None,final_scale=1):
        ''' Initialize a scaled cylinder, where the cross-sectional geometry along the object's 
            length is a scaled version of the initial geometry. 

            Args:
            xy -- Initial location of the center of the cylinder at z=0.
            r -- initial cylinder radius
            z_ex -- cylinder length
            n -- refractive index of cylinder
            nb -- background index (required for anti-aliasing)

            z_offset -- offset that sets the z-coord for the cylinder's front
            scale_func -- optional custom function. Should take in z and return a scale value. 
                          set to None to use a linear scale function, where the scale factor 
                          of the back end is set by ...
            final_scale -- the scale of the final cross-section geometry, 
                           relative to the initial geoemtry.
        '''

        super().__init__(n)
        self.p1 = p1 = [xy[0],xy[1],z_offset]
        self.p2 = [p1[0]*final_scale,p1[1]*final_scale,z_ex+z_offset]
        self.r = r
        self.rsq = r*r
        self.nb2 = nb*nb
        self.n2 = n*n
        self.z_ex = z_ex
        self.z_offset = z_offset

        def linear_func(_min,_max):
            slope =  (_max - _min)/self.z_ex
            return lambda z: slope * (z - self.z_offset) + _min

        if scale_func is None:
            scale_func = linear_func(1,final_scale)
        self.scale_func = scale_func
        self.xoffset_func = linear_func(p1[0],self.p2[0])
        self.yoffset_func = linear_func(p1[1],self.p2[1])

    def _contains(self,x,y,z):
        if not (self.z_offset <= z <= self.z_offset+self.z_ex):
            return False

        xdist = x - self.xoffset_func(z)
        ydist = y - self.yoffset_func(z)
        scale = self.scale_func(z)
        return (xdist*xdist + ydist*ydist <= scale*scale*self.rsq)

    def _bbox(self,z):
        xc = self.xoffset_func(z)
        yc = self.yoffset_func(z)
        scale = self.scale_func(z)
        xmax = xc+scale*self.r
        xmin = xc-scale*self.r
        ymax = yc+scale*self.r
        ymin = yc-scale*self.r
        return (xmin,xmax,ymin,ymax)

    def set_IORsq(self,out,z,coeff=1):
        '''overwrite base function to incorporate anti-aliasing and improve convergence'''
        if not (self.z_offset <= z <= self.z_offset+self.z_ex):
            return

        center = (self.xoffset_func(z),self.yoffset_func(z))
        scale = self.scale_func(z)
        bbox, bboxh = self.bbox_idx(z)  
        xg = self.xymesh.xg[bbox]
        yg = self.xymesh.yg[bbox]
        xhg = self.xymesh.xhg[bboxh]
        yhg = self.xymesh.yhg[bboxh]
        if xhg.shape != yhg.shape:
            print("about to error")
            print(self.xymesh.xhg.shape, self.xymesh.yhg.shape, bboxh)
        AA_circle_nonu(out,xg,yg,xhg,yhg,center,self.r*scale,self.nb2*coeff,self.n2*coeff,bbox,self.rxg,self.ryg,self.dxg,self.dyg)
    
class OpticSys(OpticPrim):
    '''base class for optical systems, collections of primitives immersed in some background medium'''
    def __init__(self,elements:List[OpticPrim], nb):
        self.elements = elements
        self.nb = nb
        self.nb2 = nb*nb
        self.vr = (
            min([x.nb2 for x in elements]),
            max([x.n for x in elements]) ** 2
        )
    
    def _bbox(self,z):
        '''default behavior. won't work if the system has disjoint pieces'''
        if len(self.elements) == 0:
            return super()._bbox(z)
        return self.elements[0]._bbox(z)
    
    def _contains(self,x,y,z):
        return self.elements[0]._contains(x,y,z)

    def set_sampling(self,xymesh:RectMesh2D):
        '''this function sets the spatial sampling for IOR computaitons'''
        super().set_sampling(xymesh)
        for elmnt in self.elements:
            elmnt.set_sampling(xymesh)

    def set_IORsq(self,out,z,coeff=1):
        '''replace values of out with IOR^2, given coordinate grids xg, yg, and z location.'''
        bbox,bboxh = self.bbox_idx(z)
        out[bbox] = self.nb2*coeff
        # with get_context("spawn").Pool(10) as p:
        #    p.map(lambda e: OpticPrim.set_IORsq(e, out, z, coeff), [self.elements])
        for elmnt in self.elements:
            elmnt.set_IORsq(out,z,coeff)

    def show_cross_section(self, z, out=None, coeff=1):
        if out is None:
            out = np.zeros(self.xymesh.shape)
        self.set_IORsq(out, z, coeff)

        plt.imshow(out, vmin=self.vr[0], vmax=self.vr[1])
        plt.show()

class Lantern(OpticSys):
    """A general photonic lantern, i.e. a number of SMF ports in a common cladding."""
    def __init__(self, port_positions, port_radii, rclad, rjack, z_ex, nvals, **kwargs):
        scale_func = kwargs.get("scale_func", None)
        final_scale = kwargs.get("final_scale", 1)
        nb = kwargs.get("nb", 1)
        ncore, nclad, njack = nvals
        elements = []
        self.core_idxs = []
        # define jack and cladding
        # if the jacket is large enough, make an optical element for it
        if rjack > rclad:
            elements.append(
                ScaledCyl([0.0, 0.0], rjack, z_ex, njack, 1, scale_func=scale_func, final_scale=final_scale)
            )
        else:
            # otherwise, use an infinite jacket
            nb = njack
            
        # cladding
        self.entry_idx = len(elements)
        elements.append(
            ScaledCyl([0.0, 0.0], rclad, z_ex, nclad, njack, scale_func=scale_func, final_scale=final_scale)
        )

        if not(type(port_radii) in (list, np.ndarray)):
            port_radii = repeat(port_radii)

        for (port, rcore) in zip(port_positions, port_radii):
            self.core_idxs.append(len(elements))
            # define SMF cores
            elements.append(
                ScaledCyl(port, rcore, z_ex, ncore, nclad, scale_func=scale_func, final_scale=final_scale)
            )

        super().__init__(elements, nb)
        self.init_core_locs = np.array(port_positions)
        self.final_core_locs = self.init_core_locs * final_scale

    def check_smfs(self, k0, verbose=True):
        """
        Checks that all the single-mode fibers are actually single-mode. We want this to return a list of all 1s.
        """

        nums_modes = map(
            lambda el: get_num_modes(k0, el.r, el.n, np.sqrt(el.nb2)), 
            map(
                lambda i: self.elements[i], 
                self.core_idxs
            )
        )

        valid = True
        for (i, n) in enumerate(nums_modes):
            if n != 1:
                if verbose:
                    print(f"Lantern setup error: port {i} is meant to support one mode but supports {n}.")
                valid = False
        
        return valid

    def check_mode_support(self, k0, verbose=True):
        """
        Checks that the multi-mode entrance supports at most as many modes as there are ports to sense them.
        """
        mmf = self.elements[self.entry_idx]
        mmf_modes = get_num_modes(k0, mmf.r, mmf.n, np.sqrt(mmf.nb2))
        valid = mmf_modes <= len(self.core_idxs)
        if not valid and verbose:
            print(f"Possible light leakage: lantern MMF entrance supports {mmf_modes} modes but can only be detected by {len(self.core_idxs)} single-mode fibers.")
        return valid


def make_lant5(offset0, *args, **kwargs):
    positions = [[0,-offset0], [-offset0,0], [0,offset0], [offset0,0], [0,0]]
    return Lantern(positions, *args, **kwargs)

def make_lant5big(offset0, *args, **kwargs):
    positions = [[0,-offset0], [-offset0,0], [0,offset0], [offset0,0], [0,0]]
    return Lantern(positions, *args, **kwargs)

def make_lant3big(offset0, *args, **kwargs):
    positions = [[np.sqrt(3)/2*offset0,-offset0/2], [-np.sqrt(3)/2*offset0,-offset0/2], [0,offset0]]
    return Lantern(positions, *args, **kwargs)

def make_lant3ms(offset0, port_radii, *args, **kwargs):
    positions = [[np.sqrt(3)/2*offset0,-offset0/2], [-np.sqrt(3)/2*offset0,-offset0/2], [0,offset0]]
    if len(port_radii) == 2:
        port_radii = [port_radii[1], port_radii[1], port_radii[0]]
    else:
        port_radii.reverse()

    return Lantern(positions, port_radii, *args, **kwargs)

def make_lant6_saval(offset0, port_radii, *args, **kwargs):
    '''6 port lantern, mode-selective, based off sergio leon-saval's paper'''
    t = 2 * np.pi / 5
    positions = [[offset0 * np.cos(i*t), offset0 * np.sin(i*t)] for i in range(5)] + [[0,0]]
    if type(port_radii) in (float, int):
        port_radii = list(repeat(port_radii, 6))
    if len(port_radii) == 4:
        port_radii = [port_radii[x] for x in [1, 1, 2, 2, 3, 0]]
    else:
        port_radii = port_radii[1:] + [port_radii[0]] # move the central SMF to the end

    return Lantern(positions, port_radii, *args, **kwargs)

def make_lant19(core_spacing, *args, **kwargs):
    positions = [[0,0]]

    for i in range(6):
        xpos = core_spacing*np.cos(i*np.pi/3)
        ypos = core_spacing*np.sin(i*np.pi/3)
        positions.append([xpos,ypos])
    
    startpos = np.array([2*core_spacing,0])
    startang = 2*np.pi/3
    pos.append(startpos)
    for i in range(11):
        if i%2==0 and i!=0:
            startang += np.pi/3
        nextpos = startpos + np.array([core_spacing*np.cos(startang),core_spacing*np.sin(startang)])
        pos.append(nextpos)
        startpos = nextpos

    return Lantern(pos, *args, **kwargs)

