"""Observables of the toy study (FLOW_NOTES.md, "Jamie's toy exercises with
OT flow matching") on full 24 x 64 canvases, around a known axis.

Regions, by the distance R of a tower centre to the (continuous) jet axis,
phi periodic: the cone R < 0.4 (Jamie's "core"), the recoil ring
0.4 <= R < 1.0, the far region R >= 1.0, and Jamie's rings for the ring-sum
comparison (edges 0, 0.1, 0.2, 0.3, 0.4, 0.6, infinity; slide 52).

Energies are summed signed (a residual readout M - B_hat can be negative).
Substructure in the cone (girth, mass, p_T^D, z_lead, z_g, R_g) needs
non-negative weights: it is computed on the image clipped at 0, a labelled
readout. Towers are massless at their centres (as substructure.py does).
"""

import numpy as np
import torch

import fm_common as fc
import toycalo as tc
from substructure import soft_drop

RINGS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.6, np.inf)
REGIONS = { 'cone' : (0.0, 0.4), 'ring' : (0.4, 1.0), 'far' : (1.0, np.inf) }

ETA_C = torch.tensor(tc.ETA_C, dtype = torch.float32)
PHI_C = torch.tensor(tc.PHI_C, dtype = torch.float32)

def offsets(axes, device):
    """(deta, dphi, dr), each (N, 24, 64), of every tower to each axis."""
    a = torch.as_tensor(np.asarray(axes, np.float32), device = device)
    deta = ETA_C.to(device)[None, :, None] - a[:, 0, None, None]
    dphi = PHI_C.to(device)[None, None, :] - a[:, 1, None, None]
    dphi = torch.remainder(dphi + np.pi, 2 * np.pi) - np.pi
    deta = deta.expand(-1, -1, tc.NPHI)
    dphi = dphi.expand(-1, tc.NETA, -1)
    return (deta, dphi, torch.sqrt(deta**2 + dphi**2))

def region_masks(dr):
    return { k : (dr >= lo) & (dr < hi) for (k, (lo, hi)) in REGIONS.items() }

@torch.no_grad()
def observables(images, axes, device, batch = 4096, substructure = True):
    """Per-image observables around the axes: region energies (signed),
    Jamie's ring sums, the cone's substructure (clipped), occupancy and the
    negative energy. Returns {name: (N,) numpy array}."""
    # pylint: disable=too-many-locals
    out = {}
    for start in range(0, len(images), batch):
        x = torch.as_tensor(np.asarray(images[start:start + batch], np.float32),
                            device = device)
        (deta, dphi, dr) = offsets(axes[start:start + batch], device)
        masks = region_masks(dr)
        row = {}
        for (k, m) in masks.items():
            row[k] = (x * m).sum((1, 2))
        row['total'] = x.sum((1, 2))
        for (lo, hi) in zip(RINGS[:-1], RINGS[1:]):
            row[f'ring_{lo:g}_{hi:g}'] = (x * ((dr >= lo) & (dr < hi))).sum((1, 2))
        row['neg_total'] = x.clamp(max = 0).sum((1, 2))
        row['neg_cone'] = (x.clamp(max = 0) * masks['cone']).sum((1, 2))
        w = x.clamp(min = 0) * masks['cone']
        e = w.sum((1, 2))
        safe = e.clamp(min = 1e-6)
        eta = ETA_C.to(device)[None, :, None].expand_as(w)
        en = (w * torch.cosh(eta)).sum((1, 2))
        px = (w * torch.cos(dphi)).sum((1, 2))
        py = (w * torch.sin(dphi)).sum((1, 2))
        pz = (w * torch.sinh(eta)).sum((1, 2))
        row['cone_pos'] = e
        row['mass'] = (en**2 - px**2 - py**2 - pz**2).clamp(min = 0).sqrt()
        row['girth'] = (w * dr).sum((1, 2)) / safe
        row['ptd'] = (w**2).sum((1, 2)).sqrt() / safe
        row['zlead'] = w.flatten(1).amax(1) / safe
        row['lead'] = w.flatten(1).amax(1)
        row['n_occ'] = (w > 0.05).sum((1, 2)).float()
        for (k, v) in row.items():
            out.setdefault(k, []).append(v.cpu().numpy())
        if substructure:
            # soft drop on the cone towers, offsets to the axis
            sel = masks['cone']
            ww = w.flatten(1).cpu().numpy().astype(np.float64)
            de = deta.flatten(1).cpu().numpy().astype(np.float64)
            dp = dphi.flatten(1).cpu().numpy().astype(np.float64)
            sm = sel.flatten(1).cpu().numpy()
            zr = np.full((len(ww), 2), np.nan)
            for k in range(len(ww)):
                m = sm[k] & (ww[k] > 0)
                zr[k] = soft_drop(ww[k][m][None], de[k][m], dp[k][m])[0]
            out.setdefault('zg', []).append(zr[:, 0])
            out.setdefault('rg', []).append(zr[:, 1])
    return { k : np.concatenate(v) for (k, v) in out.items() }

def cone_sums_everywhere(images, device, radius = 0.4):
    """The energy of the R = radius cone around every tower centre (eta does
    not wrap, phi does): (N, 24, 64). For Jamie's fake-jet definition (the
    largest R = 0.4 cone of an image)."""
    k = fc.ev.cone_kernel(radius)
    x = torch.as_tensor(np.asarray(images, np.float32), device = device)
    return fc.ev.cone_sums(x, k)
