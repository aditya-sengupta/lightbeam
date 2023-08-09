import numpy as np
from numpy import logical_not as NOT, logical_and as AND, logical_or as OR
from numba import njit
<<<<<<< HEAD
from lightbeamrs import intarea
=======
from .lightbeamrs import oneside as _oneside

def intarea(xc, yc, r, x0, x1, y0, y1):
    """
    Compute the area of overlap of a circle and a rectangle.
      xc, yc  :  Center of the circle.
      r       :  Radius of the circle.
      x0, y0  :  Corner of the rectangle.
      x1, y1  :  Opposite corner of the rectangle.
    """
    x0 = x0 - xc
    y0 = y0 - yc
    x1 = x1 - xc
    y1 = y1 - yc
    return _oneside(x1, y0, y1, r) + _oneside(y1, -x1, -x0, r) + \
           _oneside(-x0, -y1, -y0, r) + _oneside(-y0, x0, x1, r)
>>>>>>> cbf5868

# @njit
def nonu_pixwt(xc,yc,r,x,y,rx,ry,dx,dy):
    area = dx*dy/4*(rx+1)*(ry+1)
    return intarea(xc,yc,r,x-0.5*dx,x+0.5*rx*dx,y-0.5*dy,y+0.5*ry*dy)/area

def AA_circle_nonu(
    out: np.ndarray, 
    xg, yg, xgh, ygh, 
    center, R, n0, n1, where,
    rxg, ryg, dxg, dyg
):
    xdif = xgh - center[0]
    ydif = ygh - center[1]
    rsqh = xdif*xdif + ydif*ydif
    out_where = out[where] # I want to make sure this doesn't copy
    mask_in, mask_b = get_masks(rsqh,R*R)
    
    out_where[mask_in] = n1
    x0, y0 = xg[mask_b], yg[mask_b]

    rx, ry = rxg[where][mask_b], ryg[where][mask_b]
    dx, dy = dxg[where][mask_b], dyg[where][mask_b]

    area = nonu_pixwt(center[0], center[1], R, x0, y0, rx, ry, dx, dy)
    newvals = n1 * area + n0 * (1-area)
    out_where[mask_b] = newvals

@njit
def get_masks(rsqh,R2):
    maskh = (rsqh <= R2)
    n_maskh = NOT(maskh)
    mask_in = AND(maskh[1:,1:], AND(maskh[1:,:-1], AND(maskh[:-1,1:],maskh[:-1,:-1]))) 
    mask_out = AND(
        n_maskh[1:,1:], 
        AND(
            n_maskh[1:,:-1], 
            AND(
                n_maskh[:-1,1:], 
                n_maskh[:-1,:-1]
            )
        )
    ) 
    mask_b = NOT(OR(mask_in,mask_out))
    
    return mask_in, mask_b

# put AA_circle and pixwt back in before merge
# I'm just removing them now to indicate they don't need to be Rust-ified