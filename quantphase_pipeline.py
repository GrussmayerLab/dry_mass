# quantphase_pipeline.py
# A small library for 3D/4D microscopy workflows with:
#   • QP retrieval (on 3D only, taken from a specified channel in 4D)
#   • Segmentation (uses all available data: 3D or 4D)
#   • Focal plane determination (from the 3D QP volume only)
#   • Dry mass conversion from QP via m = (λ / (2π α)) ∬ φ(x,y) dx dy
#
# Conventions:
#   • Shapes:
#       3D volumes: (z, y, x)
#       4D volumes: (c, z, y, x)   # c == channel axis
#   • 'channel_bf' is the integer index into the channel axis for brightfield.
#   • Units:
#       wavelength [meters], alpha [m^3/kg] (specific RI increment),
#       pixel_size [meters/pixel] for x and y (assumed square pixels).

from dataclasses import dataclass, field
from typing import Callable, Union, Optional, Dict, Tuple
import numpy as np
import os
import pandas as pd
import tifffile as tf
import skimage.io as skio
import skimage.measure as skim
import skimage as ski
import scipy.ndimage as ndi
from tpr import phase_structure
from tpr import TopographicPhaseRetrieval as ttpr
import matplotlib.pyplot as plt
try:
    from cellpose import models, core, io, plot
except ImportError:
    raise ImportError("Cellpose is not installed. Please install it to use Cellpose segmentation.")

ArrayLike = Union[np.ndarray]

# ----------------------------- Utils: shape handling -----------------------------

def ensure_3d(a: np.ndarray) -> np.ndarray:
    """Ensure array is 3D (z,y,x)."""
    if a.ndim != 3:
        raise ValueError(f"Expected 3D (z,y,x) array, got shape {a.shape}")
    return a

def ensure_4d(a: np.ndarray) -> np.ndarray:
    """Ensure array is 4D (c,z,y,x). If 3D, add a singleton channel axis as c=1.
    Returns a view/copy in shape (c,z,y,x).
    """
    if a.ndim == 3:
        return a[None, ...]
    if a.ndim != 4:
        raise ValueError(f"Expected 3D or 4D array, got shape {a.shape}")
    return a


def _best_perm_to_match(src_shape, tgt_shape):
    """Find axis permutation of a 3D tensor that best matches target 3D shape."""
    import itertools
    perms = list(itertools.permutations((0,1,2)))
    best = None
    best_score = 1e18
    for p in perms:
        s = tuple(src_shape[i] for i in p)
        score = sum(abs(s[i] - tgt_shape[i]) for i in range(3))
        if score < best_score:
            best = p
            best_score = score
    return best

def _pad_or_crop_to_match(arr, target_shape, pad_value=0.0):
    """
    Center-pad (with pad_value) or center-crop a 3D array (z,y,x) to target_shape.
    Returns a new array with exact target_shape.
    """
    z_t, y_t, x_t = target_shape
    z, y, x = arr.shape
    out = np.full((z_t, y_t, x_t), pad_value, dtype=arr.dtype)

    # Compute source region to copy (centered overlap)
    z0 = max(0, (z - z_t)//2); z1 = z0 + min(z, z_t)
    y0 = max(0, (y - y_t)//2); y1 = y0 + min(y, y_t)
    x0 = max(0, (x - x_t)//2); x1 = x0 + min(x, x_t)

    # Compute target placement
    Z0 = max(0, (z_t - z)//2); Z1 = Z0 + (z1 - z0)
    Y0 = max(0, (y_t - y)//2); Y1 = Y0 + (y1 - y0)
    X0 = max(0, (x_t - x)//2); X1 = X0 + (x1 - x0)

    out[Z0:Z1, Y0:Y1, X0:X1] = arr[z0:z1, y0:y1, x0:x1]
    return out

def adjust_qp_to_fluorescence(phi_3d: np.ndarray,
                              fl_channels: np.ndarray,
                              invert_stack: bool = False) -> np.ndarray:
    """
    Make phi_3d (reloaded QP) match fluorescence in shape and axis order.
    - Reorder axes if needed.
    - Optionally flip z if invert_stack was used during QP retrieval.
    - Center-pad or center-crop to match (z,y,x) of fluorescence.
    """
    if fl_channels is None:
        return phi_3d

    # fluorescence expected (z,y,x,c)
    tgt = fl_channels.shape[:3]
    src = phi_3d.shape
    if len(src) != 3:
        raise ValueError(f"Expected 3D phi, got {src}")

    # 1) choose best axis permutation to match target shape (robust to bad loads)
    perm = _best_perm_to_match(src, tgt)
    if perm != (0,1,2):
        phi_3d = np.transpose(phi_3d, perm)

    # 2) optional flip along z if requested
    if invert_stack:
        phi_3d = phi_3d[::-1, :, :]

    # 3) pad/crop to exact target (use zeros to keep Cellpose happy)
    phi_3d = _pad_or_crop_to_match(phi_3d, tgt, pad_value=0.0)
    return phi_3d

# ----------------------------- QP retrieval (3D only) ----------------------------

def qp_retrieve_3d_from_channel(
    data: np.ndarray,
    channel_bf: int,
    phase_str: phase_structure = phase_structure(),
    compute_qp: bool = True,
    invert_stack: bool = False
) -> np.ndarray:
    """Retrieve a 3D QP volume (z,y,x) from input data.

    - If data is 4D (c,z,y,x), we extract BF as (z,y,x) and pass ONLY BF through QP.
      Fluorescence channels are returned untouched as (z,y,x,c).
    - If data is 3D (z,y,x), we use it directly and fluorescence is None.
    """
    if data.ndim == 4:
        c, z, y, x = data.shape
        if not (0 <= channel_bf < c):
            raise IndexError(f"channel_bf={channel_bf} out of range for c={c}")

        # Brightfield stack for QP: (z,y,x)
        vol3 = data[channel_bf, ...]  # (z,y,x)

        # Fluorescence channels (all except BF), returned as (z,y,x,c_fl)
        if c > 1:
            fl = np.delete(data, channel_bf, axis=0)     # (c_fl, z, y, x)
            fl_channels = np.moveaxis(fl, 0, -1)         # (z, y, x, c_fl)
        else:
            fl_channels = None

    elif data.ndim == 3:
        vol3 = data
        fl_channels = None
    else:
        raise ValueError(f"Expected 3D or 4D input, got shape {data.shape}")

    if compute_qp:
        vol3 = ensure_3d(vol3)                 # (z,y,x)
        proc = ttpr()
        # ttpr expects (y,x,z) -> compute -> back to (z,y,x)
        vol3_yxz = np.transpose(vol3, (1, 2, 0))  # (y,x,z)
        if invert_stack:
            vol3_yxz = vol3_yxz[:, :, ::-1]
        phi_yxz, _ = proc.getQP(stack=vol3_yxz, struct=phase_str)
        phi = np.transpose(phi_yxz, (2, 0, 1))    # (z,y,x)
    else:
        phi = None

    return phi, fl_channels


def prepare_stack_for_qp(data: np.ndarray, invert_stack: bool = False) -> np.ndarray:
    """Prepare input data for QP retrieval.
    Input (z,y,x) -> output (y,x,z) cropped (and optionally inverted in z).
    """
    data = np.transpose(data, (1,2,0))  # (y,x,z)
    print(f"Stack shape after permutation: {data.shape}")
    p = ttpr()
    data = p.cropXY(input_=data)        # (y,x,z)
    print(f"Stack shape after cropping: {data.shape}")
    if invert_stack:
        data = data[:, :, ::-1]
        print('Inverted stack for QP retrieval')
    return data

# ----------------------------- Segmentation (3D/4D) ------------------------------

def filter_masks_by_size(mask: np.ndarray, min_size: int = 64) -> np.ndarray:
    """Remove small objects from a segmentation mask."""
    label_mask = skim.label(mask)
    props = skim.regionprops(label_mask)
    clean_mask = np.zeros_like(mask, dtype=int)
    for prop in props:
        if prop.area >= min_size:
            clean_mask[label_mask == prop.label] = prop.label
    return clean_mask
# projection function
def _projection_for_segmentation(data: np.ndarray, method: str = "focal", fp: int = 0) -> np.ndarray:
    """
    Reduce a (z, y, x) stack to a 2D (y, x) image.
    If you ever pass 4D (z, y, x, c), call this per-channel before stacking.
    """
    if data.ndim != 3:
        raise ValueError(f"_projection_for_segmentation expects 3D (z,y,x); got {data.shape}")

    if method == "focal":
        return data[:, :, fp]  # (y, x)
    elif method == "mean":
        return data.mean(axis=-1)
    elif method == "max":
        return data.max(axis=-1)
    else:
        raise ValueError("method must be 'focal', 'mean', or 'max'")



def setup_cellpose_model(model_type: str = "cyto"):
    """Set up Cellpose model for segmentation."""
    io.logger_setup()  # progress printing
    use_GPU = core.use_gpu()
    yn = ['NO', 'YES']
    print(f'>>> GPU activated? {yn[use_GPU]}')
    model = models.CellposeModel(model_type=model_type, gpu=use_GPU)
    return model

def segment_volume(
    data: np.ndarray,
    method: str = "focal",
    model_type: str = "cyto",
    model = None,
    fp: int = 0,
    fluorescence_channels: np.ndarray = None,
    flow_threshold: float = 0.4,
    cell_threshold: float = 0.0
) -> np.ndarray:
    """Segment the sample using all available data.
    - Primary image comes from QP stack reduced by `method` (usually 'focal').
    - If fluorescence is provided, it must be (z,y,x,c) and we add each channel
      as an auxiliary 2D image using the same reduction.
    Returns: mask (2D), the base image used, flow, and a figure.
    """
    base_img_2d = _projection_for_segmentation(data, method=method, fp=fp)  # (y,x)
    img_list = [base_img_2d]

    # fluorescence_channels expected as (z,y,x,c)
    if fluorescence_channels is not None and fluorescence_channels.ndim == 4:
        num_fl = fluorescence_channels.shape[-2]
        for ch in range(num_fl):
            fl3 = fluorescence_channels[:,:,ch,:]            # (z,y,x)
            img_list.append(_projection_for_segmentation(fl3, method=method, fp=fp))
    elif fluorescence_channels is not None:
        raise ValueError(f"fluorescence_channels must be (z,y,x,c), got {fluorescence_channels.shape}")

    if model is None:
        model = setup_cellpose_model(model_type=model_type)

    imgs_np = np.array(img_list)  # (n_images, y, x)
    mask, flow, style = model.eval(
        imgs_np,
        batch_size=32,
        flow_threshold=flow_threshold,
        cellprob_threshold=cell_threshold,
        do_3D=False,
    )

    fig = plt.figure(figsize=(12,5))
    plot.show_segmentation(fig, base_img_2d, mask, flow[0])
    plt.tight_layout()
    plt.show()

    return mask, base_img_2d, flow, fig


def segment_nucleus(
    data: np.ndarray, # nuclear data only
    method: str = "focal",
    fp: int = 0,
) -> np.ndarray:
    """Simple nucleus segmentation on a focal slice."""
    img = [_projection_for_segmentation(data, method=method, fp=fp)][0]
    mask = get_nucleus_mask_from_image(img)

    fig, ax= plt.subplots(1,3, figsize=(12,4))
    ax[0].imshow(img, cmap='gray'); ax[0].set_title('Nucleus input')
    image_label_overlay = ski.color.label2rgb(mask, image=img, bg_label=0)
    ax[1].imshow(image_label_overlay); ax[1].set_title('Overlay')
    ax[2].imshow(mask); ax[2].set_title('Labels')
    for a in ax: a.axis('off')
    plt.tight_layout()
    plt.show()
    return mask, img, fig

# ------------------------- Focal plane determination (QP) ------------------------

def compute_fourier_sharpness(image, high_freq_cutoff=0.1):
    """
    Compute a sharpness score from the high-frequency content of an image using Fourier analysis.
    """
    image = image.astype(np.float32)
    image -= image.mean()
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    radius = np.sqrt((X - ccol)**2 + (Y - crow)**2)
    mask = radius > (min(crow, ccol) * high_freq_cutoff)
    high_freq_energy = np.sum((magnitude[mask])**2)
    return high_freq_energy

def determine_focal_plane_from_qp(phi_3d: np.ndarray) -> int:
    """Determine focal plane (z-index) from a 3D QP volume using a focus metric."""
    phi_3d = ensure_3d(phi_3d)
    scores = [compute_fourier_sharpness(phi_3d[:,:,k]) for k in range(phi_3d.shape[2])]
    z_star = int(np.argmax(scores))
    return z_star

# ------------------------- Dry mass conversion from QP ---------------------------

def dry_mass_from_qp(
    phi_2d: np.ndarray,
    wavelength_m: float,
    alpha_m3_per_g: float,
    pixel_size_m: float,
) -> float:
    """Compute dry mass (kg) from a single 2D QP map φ(x,y) using:
        m = (λ / (2π α)) ∬ φ(x,y) dx dy
    with dx=dy=pixel_size_m. Assumes square pixels.
    """
    phi = phi_2d.astype(np.float64)
    opd = phi*(wavelength_m / (2.0 * np.pi))  # in meters
    area_element = pixel_size_m * pixel_size_m
    m = np.nansum(opd * area_element) / alpha_m3_per_g
    return float(m)

def separate_masks(phi: np.ndarray, mask: np.ndarray, plot_flag = False) -> list:
    m=(mask>0).astype(np.int8)
    img_masked = phi * m
    img_masked[img_masked==0] = 0.000
    if plot_flag:
        fig, ax = plt.subplots(1,2, figsize=(10,5))
        ax[0].imshow(phi, cmap='gray'); ax[0].set_title('Original QP slice')
        ax[1].imshow(img_masked, cmap='gray'); ax[1].set_title('Masked QP slice')
        plt.show()
    label = skim.label(mask)
    props = skim.regionprops(label, intensity_image=img_masked)
    return props

def dry_mass_per_cell(
    phi_fp: np.ndarray,
    mask: np.ndarray,
    wavelength_m: float,
    alpha_m3_per_g: float,
    pixel_size_m: float,
) -> dict:
    """Compute dry mass per cell and return a dict with visualisation, area and mass info per cell."""
    cell_masks = separate_masks(phi_fp, mask)
    cell = {}
    pixel_area = pixel_size_m**2
    for ii, p in enumerate(cell_masks):
        print(f"Processing region {ii+1}/{len(cell_masks)}")
        min_phase = np.min(p.image_intensity)
        cell_data = p.image_intensity - min_phase
        area = p.area_filled * pixel_area
        mass = dry_mass_from_qp(cell_data, wavelength_m, alpha_m3_per_g, pixel_size_m)
        cell[ii] = {}
        cell[ii]['mass'] = mass
        cell[ii]['area'] = area
        fig, ax = plt.subplots(dpi=100, constrained_layout=True)
        im = ax.imshow(cell_data, cmap='inferno', interpolation="nearest")
        ax.set_title(f"Region {ii+1} - Mass: {cell[ii]['mass']*10**(12):.4f} pg")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_frame_on(False)
        cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.9)
        cbar.set_label("phase [rad]", rotation=270, labelpad=12)
        cell[ii]['figure'] = fig
        cell[ii]['data'] = cell_data
    return cell

def dry_mass_nuc_per_cell(
    phi_fp: np.ndarray,
    mask_cell: np.ndarray,
    mask_nuc: np.ndarray,
    wavelength_m: float,
    alpha_m3_per_g: float,
    pixel_size_m: float,
) -> dict:
    """(Kept for compatibility) Compute dry mass per nucleus using matching cell-baseline."""
    cell_masks = separate_masks(phi_fp, mask_cell)
    cell_nuc = separate_masks(phi_fp, mask_nuc)
    cell = {}
    pixel_area = pixel_size_m**2
    label_map = {c.label: c for c in cell_masks}
    for ii, p in enumerate(cell_nuc):
        print(f"Processing region {ii+1}/{len(cell_nuc)}")
        matching_cell = label_map.get(p.label)
        if matching_cell is None:
            min_phase = 0.0
        else:
            min_phase = np.min(matching_cell.image_intensity)
        nuc_data = p.image_intensity - min_phase
        area = p.area_filled * pixel_area
        mass = dry_mass_from_qp(nuc_data, wavelength_m, alpha_m3_per_g, pixel_size_m)
        cell[ii] = {}
        cell[ii]['mass'] = mass
        cell[ii]['area'] = area
        fig, ax = plt.subplots(dpi=100, constrained_layout=True)
        im = ax.imshow(nuc_data, cmap='inferno', interpolation="nearest")
        ax.set_title(f"Region {ii+1} - Mass: {cell[ii]['mass']*10**(12):.4f} pg")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_frame_on(False)
        cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.9)
        cbar.set_label("phase [rad]", rotation=270, labelpad=12)
        cell[ii]['figure'] = fig
        cell[ii]['data'] = nuc_data
    return cell

# ------------------------- Per-cell class + metrics (NEW) ------------------------

@dataclass
class Cell:
    label: int
    centroid: Tuple[float, float]
    # binary masks (2D, same shape as phi_fp)
    mask_cell: np.ndarray
    mask_nuc: np.ndarray
    mask_cyto: np.ndarray
    # areas in m^2
    area_cell: float = 0.0
    area_nuc: float = 0.0
    area_cyto: float = 0.0
    # masses in kg
    mass_cell: float = 0.0
    mass_nuc: float = 0.0
    mass_cyto: float = 0.0
    # optional visuals
    fig_cell: Optional[plt.Figure] = field(default=None, repr=False)
    fig_nuc: Optional[plt.Figure]  = field(default=None, repr=False)
    fig_cyto: Optional[plt.Figure] = field(default=None, repr=False)

def _compute_area_m2(mask: np.ndarray, pixel_size_m: float) -> float:
    return float(mask.sum()) * (pixel_size_m ** 2)

def _compute_mass_kg_from_mask(
    phi_2d: np.ndarray,
    mask: np.ndarray,
    wavelength_m: float,
    alpha_m3_per_g: float,
    pixel_size_m: float,
    baseline_phase: float = 0.0,
) -> float:
    phi_corr = np.where(mask, phi_2d - baseline_phase, 0.0)
    return dry_mass_from_qp(phi_corr, wavelength_m, alpha_m3_per_g, pixel_size_m)

def build_cells_and_metrics(
    phi_fp: np.ndarray,
    cell_mask: np.ndarray,
    nuc_mask: np.ndarray,
    wavelength_m: float,
    alpha_m3_per_g: float,
    pixel_size_m: float,
    make_figures: bool = True,
) -> Dict[int, Cell]:
    cells: Dict[int, Cell] = {}
    props = ski.measure.regionprops(cell_mask)
    for p in props:
        L = int(p.label)
        mask_cell = (cell_mask == L)
        mask_nuc  = (nuc_mask  == L)
        mask_cyto = mask_cell & (~mask_nuc)
        cell_phase_vals = phi_fp[mask_cell]
        baseline = float(np.min(cell_phase_vals)) if cell_phase_vals.size else 0.0
        area_cell = _compute_area_m2(mask_cell, pixel_size_m)
        area_nuc  = _compute_area_m2(mask_nuc,  pixel_size_m)
        area_cyto = _compute_area_m2(mask_cyto, pixel_size_m)
        mass_cell = _compute_mass_kg_from_mask(phi_fp, mask_cell, wavelength_m, alpha_m3_per_g, pixel_size_m, baseline)
        mass_nuc  = _compute_mass_kg_from_mask(phi_fp, mask_nuc,  wavelength_m, alpha_m3_per_g, pixel_size_m, baseline)
        mass_cyto = _compute_mass_kg_from_mask(phi_fp, mask_cyto, wavelength_m, alpha_m3_per_g, pixel_size_m, baseline)
        cell = Cell(
            label=L,
            centroid=(float(p.centroid[0]), float(p.centroid[1])),
            mask_cell=mask_cell.astype(bool),
            mask_nuc=mask_nuc.astype(bool),
            mask_cyto=mask_cyto.astype(bool),
            area_cell=area_cell,
            area_nuc=area_nuc,
            area_cyto=area_cyto,
            mass_cell=mass_cell,
            mass_nuc=mass_nuc,
            mass_cyto=mass_cyto,
        )
        if make_figures:
            def _make_fig(mask, title):
                fig, ax = plt.subplots(dpi=100, constrained_layout=True)
                data = np.where(mask, phi_fp - baseline, np.nan)
                im = ax.imshow(data, cmap='inferno', interpolation="nearest")
                ax.set_title(title)
                ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)
                cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.9)
                cbar.set_label("phase [rad]", rotation=270, labelpad=12)
                return fig
            cell.fig_cell = _make_fig(mask_cell, f"Cell {L} - Mass: {mass_cell*1e12:.4f} pg")
            if mask_nuc.any():
                cell.fig_nuc  = _make_fig(mask_nuc,  f"Cell {L} nucleus - Mass: {mass_nuc*1e12:.4f} pg")
            if mask_cyto.any():
                cell.fig_cyto = _make_fig(mask_cyto, f"Cell {L} cytosol - Mass: {mass_cyto*1e12:.4f} pg")
        cells[L] = cell
    return cells

# ------------------------- nucleus helpers -----------------------

def get_nucleus_mask_from_image(
    image: np.ndarray, area_threshold: int = 100, nuc_distance: int = 25):
    image = ski.filters.gaussian(image, sigma=3)
    thresh = ski.filters.threshold_otsu(image)
    binary = image > thresh
    binary  = ski.morphology.binary_erosion(binary)

    distance = ndi.distance_transform_edt(binary)
    coords = ski.feature.peak_local_max(distance, footprint=np.ones((7, 7)),
                                        min_distance=nuc_distance, labels=binary)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = ski.segmentation.watershed(-distance, markers, mask=binary)
    labels = ski.segmentation.expand_labels(labels, distance=1)
    label_props = ski.measure.regionprops(labels)
    for l in label_props:
        if l.area < area_threshold:
            labels[labels==l.label] = 0
    return labels

def link_nucleus_to_cells(nuc_mask: np.ndarray, cell_mask: np.ndarray) -> np.ndarray:
    nmask_props = ski.measure.regionprops(nuc_mask)
    cmask_props = ski.measure.regionprops(cell_mask)
    nmask_new  = np.zeros_like(nuc_mask)
    for ii, nprop in enumerate(nmask_props):
        cell_idx = cell_mask[int(nprop.centroid[0]), int(nprop.centroid[1])]
        nmask_new[nuc_mask==nprop.label] = cell_idx
    return nmask_new

# ------------------------- I/O helpers for QP stacks -----------------------------

def write_qp_stack_to_tiff(phi_3d: np.ndarray, path_save: str, filename: str, s=None):
    """Write 3D QP stack to a multi-page TIFF file."""
    if len(phi_3d.shape) == 4:
        phi_3d = np.transpose(phi_3d, (2,1,0,3))
        print("4d")
        axes = 'ZYXT'
    else:
        phi_3d = np.transpose(phi_3d, (2,1,0))
        print("3d")
        axes = "ZXY"
    outputImageFileName = os.path.join(path_save, filename)
    tf.imwrite(
        outputImageFileName,
        phi_3d,
        resolution=(1/s.optics_dx, 1/s.optics_dx),
        metadata={
            'spacing': s.optics_dz,
            'unit': 'um',
            'finterval': 1,
            'axes': axes
        })
    print(f"Saved QP stack to {outputImageFileName}")

import json
def write_phase_structure(s, path, filename=None):
    outputImageFileName = os.path.join(path, f"phase_structure.json")
    s.filename = filename
    with open(outputImageFileName, 'w') as fp:
        json.dump( s.__dict__ , fp)

def load_data(filename):
    data = skio.imread(filename)
    # fix logical check (use 'and', not bitwise '&')
    if (data.ndim >= 3) and (data.shape[0] != data.shape[1] and data.shape[1] == data.shape[2]):
        data = np.transpose(data, (1,2,0))
        print(f"Updated shape: {data.shape}")
    data = np.swapaxes(data,0,1)
    return data

def write_cell_images(cell_dict, path_save, filename):
    """Write individual cell images to files."""
    if not os.path.exists(path_save):
        os.mkdir(path_save)
        print("Created %s " % path_save)
    for cell_id, cell_info in cell_dict.items():
        fig = cell_info['figure']
        filename_png = os.path.join(path_save, f"{filename}_cell_{cell_id+1}.png")
        fig.savefig(filename_png)
        filename_tiff = filename_png.replace('.png', '.tiff')
        skio.imsave(filename_tiff, cell_info['data'].astype(np.float32), check_contrast=False)
        plt.close(fig)
        print(f"Saved cell image to {filename_png}")

def write_mask(img: np.ndarray, mask: np.ndarray, flow: np.ndarray, path_save: str, filename: str):
    """Write segmentation mask to a TIFF/PNG via Cellpose helper."""
    if not os.path.exists(path_save):
        os.mkdir(path_save)
        print("Created %s " % path_save)
    io.save_masks(img, mask, flow, file_names=os.path.join(path_save, filename))

def write_dataframe(cell: dict, path_save: str, filename: str):
    """Write dry mass data to a CSV file."""
    if not os.path.exists(path_save):
        os.mkdir(path_save)
        print("Created %s " % path_save)
    if any(cell.items()):
        outkeys = [k for k in cell[0].keys() if k not in ['figure', 'data']]
        df = pd.DataFrame([{k: cell[ii][k] for k in outkeys} for ii in range(len(cell))])
        outfname = os.path.join(path_save, f"{filename}_cell_data.csv")
        df.to_csv(outfname, index=True)
        print(f"Saved dry mass data to {outfname}")

def write_segmentation(fig, path_save: str, filename: str):
    """Write segmentation figure to file."""
    if not os.path.exists(path_save):
        os.mkdir(path_save)
        print("Created %s " % path_save)
    outfname = os.path.join(path_save, f"{filename}_segmentation.png")
    fig.savefig(outfname)
    print(f"Saved segmentation figure to {outfname}")

# ------------------------- NEW writers for Cell objects --------------------------

def write_cells_dataframe(cells: Dict[int, Cell], path_save: str, filename: str):
    """Write per-cell nucleus/cytosol/whole metrics to CSV."""
    if not os.path.exists(path_save):
        os.mkdir(path_save)
    if not cells:
        return
    rows = []
    for L, c in cells.items():
        rows.append({
            "label": L,
            "centroid_row": c.centroid[0],
            "centroid_col": c.centroid[1],
            "area_cell_m2": c.area_cell,
            "area_nuc_m2":  c.area_nuc,
            "area_cyto_m2": c.area_cyto,
            "mass_cell_kg": c.mass_cell,
            "mass_nuc_kg":  c.mass_nuc,
            "mass_cyto_kg": c.mass_cyto,
            "mass_cell_pg": c.mass_cell * 1e12,
            "mass_nuc_pg":  c.mass_nuc  * 1e12,
            "mass_cyto_pg": c.mass_cyto * 1e12,
        })
    df = pd.DataFrame(rows).sort_values("label").reset_index(drop=True)
    outfname = os.path.join(path_save, f"{filename}_cells.csv")
    df.to_csv(outfname, index=False)
    print(f"Saved per-cell metrics to {outfname}")

def write_cell_component_images(cells: Dict[int, Cell], path_save: str, filename: str):
    """Save cell, nucleus, cytosol figures if present (like your write_cell_images)."""
    if not os.path.exists(path_save):
        os.mkdir(path_save)
    for L, c in cells.items():
        if c.fig_cell is not None:
            fp = os.path.join(path_save, f"{filename}_cell_{L}.png")
            c.fig_cell.savefig(fp); plt.close(c.fig_cell)
            print(f"Saved {fp}")
        if c.fig_nuc is not None:
            fp = os.path.join(path_save, f"{filename}_cell_{L}_nucleus.png")
            c.fig_nuc.savefig(fp); plt.close(c.fig_nuc)
            print(f"Saved {fp}")
        if c.fig_cyto is not None:
            fp = os.path.join(path_save, f"{filename}_cell_{L}_cytosol.png")
            c.fig_cyto.savefig(fp); plt.close(c.fig_cyto)
            print(f"Saved {fp}")

# ------------------------- High-level convenience wrapper ------------------------

@dataclass
class PipelineConfig:
    channel_bf: int = 0              # index into c-axis for BF channel (when input is 4D)
    seg_reduce: str = "focal"        # how to collapse z for segmentation: 'mean' or 'max' or 'focal'
    seg_reduce_nuc: str = "max"        # how to collapse z for nucleus segmentation: 'mean' or 'max' or 'focal'
    opening_radius: int = 1
    min_size: int = 64
    wavelength_m: float = 550e-9     # default 550 nm
    alpha_m3_per_g: float = 0.18e-3  # typical for proteins (mL/g -> m^3/kg); adjust for your sample
    pixel_size_m: float = 0.1e-6     # 100 nm pixels; adjust per system
    ps: phase_structure = phase_structure()  # TPR structure parameters
    fp: int = -1                      # focal plane index for segmentation when method='focal'
    model_type: str = "cyto"          # Cellpose model type for segmentation
    flow_threshold: float = 0.4
    cell_threshold: float = 0.0
    overwrite_qp: bool = False        # whether to overwrite existing QP results
    cellpose_model = None             # pre-loaded Cellpose model (optional)
    invert_stack: bool = False        # whether to invert the stack for QP retrieval
    segment_nucleus : bool = False    # whether to segment nucleus
    channel_nuc: int = 1              # channel index for nucleus segmentation (if enabled)
    cell_area_threshold: int = 6400   # pixel-wise area threshold

@dataclass
class PipelineOutputs:
    phi_3d: np.ndarray           # (z,y,x) QP volume from BF
    seg_mask: np.ndarray         # (2D) boolean/int mask at focal plane
    z_focal: int                 # best focal plane index
    dry_masses: dict             # legacy per-cell dict (whole-cell)
    cells: Optional[Dict[int, Cell]] = None  # NEW: rich per-cell results

def plot_segmentation_results(
    img: np.ndarray,
    mask: np.ndarray,
    flow: np.ndarray
):
    fig = plt.figure(figsize=(12,5))
    plot.show_segmentation(fig, img, mask, flow[0])
    plt.tight_layout()
    plt.show()
    return fig

def run_pipeline(
    data: np.ndarray,
    cfg: PipelineConfig,
    path_save: str = os.getcwd(),
    filename: str = None,
) -> PipelineOutputs:
    """Run the full workflow:
        1) QP retrieval on 3D data only (from BF channel or the 3D input)
        2) Segmentation using all available data (3D or 4D)
        3) Focal plane determination from the 3D QP volume
        4) Dry mass conversion from QP (per-slice + at focal plane)
        5) (NEW) Per-cell nucleus/cytosol/whole-cell metrics via Cell class
    """
    # 1) QP retrieval, conditional caching
    fname_modifier = 'qp_'
    file_out = fname_modifier+filename if filename is not None else 'qp_dummy.tif'
    filepath = os.path.join(path_save, file_out)

    try:
        file_id = filename.split('.')[0]
    except:
        file_id = 'dummy'

    if not os.path.exists(path_save):
        os.mkdir(path_save)
        print("Created %s " % path_save)

    if not os.path.exists(filepath) or cfg.overwrite_qp:
        phi_3d, fl_channels = qp_retrieve_3d_from_channel(
            data=data, channel_bf=cfg.channel_bf,
            phase_str=cfg.ps,
            compute_qp=True,
            invert_stack=cfg.invert_stack
        )
        write_qp_stack_to_tiff(phi_3d, path_save, file_out, cfg.ps)
        write_phase_structure(cfg.ps, path_save, filename)
        phi_3d = load_data(filepath)
        #phi_3d = adjust_qp_to_fluorescence(phi_3d, fl_channels, invert_stack=cfg.invert_stack)

    else:
        _, fl_channels = qp_retrieve_3d_from_channel(
            data=data, channel_bf=cfg.channel_bf,
            phase_str=cfg.ps,
            compute_qp=False,
            invert_stack=cfg.invert_stack
        )
        phi_3d = load_data(filepath)
        #phi_3d = adjust_qp_to_fluorescence(phi_3d, fl_channels, invert_stack=cfg.invert_stack)

    # cut out brightfield data from fl_channels
    fl_channels = fl_channels[:,:,1:,:] 
    # 2) Focal plane from QP
    if cfg.fp >= 0:
        z_focal = cfg.fp
    else:
        z_focal = determine_focal_plane_from_qp(phi_3d)

    # 3) Segmentation using all available data
    model = cfg.cellpose_model if cfg.cellpose_model is not None else setup_cellpose_model(model_type=cfg.model_type)

    seg_mask, base_img_for_vis, flow, fig = segment_volume(
        data=phi_3d, method=cfg.seg_reduce, 
        model_type=cfg.model_type,
        model=model,
        fp=z_focal, 
        fluorescence_channels=fl_channels, 
        flow_threshold=cfg.flow_threshold,
        cell_threshold=cfg.cell_threshold
    )

    # reduce the phase image to the projection/focal plane 
    phi_fp = phi_3d[:, :, z_focal]

        # Sanity: shapes must match for per-pixel integration
    if phi_fp.shape != seg_mask.shape:
        # try the common transpose mismatch once
        if phi_fp.T.shape == seg_mask.shape:
            phi_fp = phi_fp.T
        else:
            raise ValueError(
                f"Shape mismatch: phi_fp {phi_fp.shape} vs seg_mask {seg_mask.shape}. "
                "Check QP axis order and fluorescence alignment."
        )


    seg_mask = filter_masks_by_size(seg_mask, min_size=cfg.cell_area_threshold)
    fig = plot_segmentation_results(phi_fp, seg_mask, flow)
    write_segmentation(fig, path_save, file_id)

    # 3.2) Optionally segment nucleus only
    nuc_mask = None
    if cfg.segment_nucleus and fl_channels is not None:
        nuc_mask, nuc_img, nuc_fig = segment_nucleus(
            data=np.squeeze(fl_channels[:,:,cfg.channel_nuc,:]),
            method=cfg.seg_reduce_nuc,
            fp=z_focal
        )
        nuc_mask = link_nucleus_to_cells(nuc_mask, seg_mask)
        write_segmentation(nuc_fig, path_save, file_id+'_nucleus')

    # 4) Dry mass conversion (legacy whole cells for backward compatibility)
    dry_masses = dry_mass_per_cell(
        phi_fp=phi_fp,
        mask=seg_mask,
        wavelength_m=cfg.wavelength_m,
        alpha_m3_per_g=cfg.alpha_m3_per_g,
        pixel_size_m=cfg.pixel_size_m,
    )
    write_cell_images(dry_masses, path_save, file_id if file_id is not None else 'output')
    write_mask(img=phi_fp, mask=seg_mask, flow=flow, path_save=path_save, filename=file_id)
    write_dataframe(dry_masses, path_save, file_id)

    # 4.2) NEW: per-cell nucleus / cytosol / whole-cell metrics
    cells = None
    if cfg.segment_nucleus and nuc_mask is not None:
        cells = build_cells_and_metrics(
            phi_fp=phi_fp,
            cell_mask=seg_mask,
            nuc_mask=nuc_mask,
            wavelength_m=cfg.wavelength_m,
            alpha_m3_per_g=cfg.alpha_m3_per_g,
            pixel_size_m=cfg.pixel_size_m,
            make_figures=True,
        )
        write_cells_dataframe(cells, path_save, file_id+'_nucleus_cytosol_cell')
        write_cell_component_images(cells, path_save, file_id+'_nucleus_cytosol_cell')

    return PipelineOutputs(
        phi_3d=phi_3d,
        seg_mask=seg_mask,
        z_focal=z_focal,
        dry_masses=dry_masses,
        cells=cells
    )
