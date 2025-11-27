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

from dataclasses import dataclass
from typing import Callable, Union
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

# ----------------------------- QP retrieval (3D only) ----------------------------


def qp_retrieve_3d_from_channel(
    data: np.ndarray,
    channel_bf: int,
    phase_str: phase_structure = phase_structure(),
    compute_qp: bool = True,
    invert_stack: bool = False
) -> np.ndarray:
    """Retrieve a 3D QP volume (z,y,x) from input data.
    - If data is 4D (c,z,y,x), we extract data[channel_bf] (shape (z,y,x)).
    - If data is 3D (z,y,x), we use it directly.
    - We then compute per-slice QP using `per_slice_fn` on each z-plane.

    Returns:
        phi: np.ndarray of shape (z,y,x)  (3D QP volume)
    """
    if data.ndim == 4:
        if not (0 <= channel_bf < data.shape[0]):
            raise IndexError(f"channel_bf={channel_bf} out of range for c={data.shape[0]}")
        data_temp = []
        for ch in range(data.shape[3]):
            data_temp.append(prepare_stack_for_qp(np.squeeze(data[:, :, :, ch]), invert_stack=invert_stack))

        #vol3 = data[:,:,:,channel_bf]
        vol3 = data_temp[channel_bf]
        data_temp.pop(channel_bf)
        fl_channels = np.array(data_temp)

    elif data.ndim == 3:
        vol3 = data
        fl_channels = None
    else:
        raise ValueError(f"Expected 3D or 4D input, got shape {data.shape}")

    if compute_qp:
        vol3 = ensure_3d(vol3)
        z, y, x = vol3.shape
        proc = ttpr()
        phi,mask = proc.getQP(stack=vol3,struct=phase_str)
    else: 
        phi = None 
    
    return phi, fl_channels 

def prepare_stack_for_qp(data: np.ndarray, invert_stack = False) -> np.ndarray:
    """Prepare input data for QP retrieval.
    """
    data = np.transpose(data, (1,2,0))
    print(f"Stack shape after permutation: {data.shape}")
    p = ttpr()
    data=p.cropXY(input_=data)
    print(f"Stack shape after cropping: {data.shape}")
    # invert the stack
    if invert_stack:
        data= data[:,:,::-1]
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


def _projection_for_segmentation(data: np.ndarray, method: str = "mean", fp: int = 0) -> np.ndarray:
    """Combine 3D or 4D data into a 3D stack (z,y,x) for segmentation.
    If 4D (c,z,y,x), reduce across channels using mean or max.
    """
    if method == "mean":
        vol3 = data.mean(axis=2)
    elif method == "max":
        vol3 = data.max(axis=2)
    elif method == "focal":
        vol3 = data[:,:,fp]
    else:
        raise ValueError("method must be 'mean' or 'max'")
    return vol3

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
    - If 4D, we first reduce across channels -> (z,y,x) via `method` (mean or max).
    - Then Otsu threshold per z-slice.
    - Morphological opening and remove small objects per z.
    Returns a boolean mask (z,y,x).
    """
    img = [_projection_for_segmentation(data, method=method, fp=fp)]

    if fluorescence_channels is not None:
        for ch in range(fluorescence_channels.shape[0]):
            img.append(_projection_for_segmentation(np.squeeze(fluorescence_channels[ch,:,:,:]),
                                                    method=method,
                                                    fp=fp))
            
    if model is None:
        model = setup_cellpose_model(model_type=model_type)

    mask, flow, style = model.eval(np.array(img), 
                                   batch_size=32,
                                    flow_threshold=flow_threshold,
                                    cellprob_threshold=cell_threshold,
                                    do_3D=False, )

    fig = plt.figure(figsize=(12,5))
    plot.show_segmentation(fig, img[0], mask, flow[0])
    plt.tight_layout()
    plt.show()

    return mask, img[0], flow, fig



def segment_nucleus(
    data: np.ndarray, # nuclear data only
    method: str = "focal",
    fp: int = 0,
) -> np.ndarray:
    """Segment the sample using all available data.
    - If 4D, we first reduce across channels -> (z,y,x) via `method` (mean or max).
    - Then Otsu threshold per z-slice.
    - Morphological opening and remove small objects per z.
    Returns a boolean mask (z,y,x).
    """
    img = [_projection_for_segmentation(data, method=method, fp=fp)][0]
    mask = get_nucleus_mask_from_image(img)

    fig, ax= plt.subplots(1,3)
    ax[0].imshow(img, cmap='gray')
    image_label_overlay = ski.color.label2rgb(mask, image=img, bg_label=0)
    ax[1].imshow(image_label_overlay)
    ax[2].imshow(mask)
    plt.tight_layout()
    plt.show()
    
    return mask, img, fig


def setup_cellpose_model(model_type: str = "cyto"):
    """Set up Cellpose model for segmentation."""
    
    io.logger_setup() # run this to get printing of progress

    #Check if notebook instance has GPU access
    use_GPU = core.use_gpu()
    yn = ['NO', 'YES']
    print(f'>>> GPU activated? {yn[use_GPU]}')

    model = models.CellposeModel(model_type=model_type, gpu=use_GPU)
    
    return model


# ------------------------- Focal plane determination (QP) ------------------------


def compute_fourier_sharpness(image, high_freq_cutoff=0.1):
    """
    Compute a sharpness score from the high-frequency content of an image using Fourier analysis.
    
    Parameters:
    - image (2D or 3D ndarray): Grayscale or RGB image.
    - high_freq_cutoff (float): Fraction (0-1) of radius to exclude low frequencies.
    
    Returns:
    - float: Sharpness score based on high-frequency energy.
    """
    # Normalize image
    image = image.astype(np.float32)
    image -= image.mean()

    # Fourier transform
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    # Create high-frequency mask
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    radius = np.sqrt((X - ccol)**2 + (Y - crow)**2)
    mask = radius > (min(crow, ccol) * high_freq_cutoff)

    # Compute high-frequency energy
    high_freq_energy = np.sum((magnitude[mask])**2)
    return high_freq_energy

def determine_focal_plane_from_qp(phi_3d: np.ndarray) -> int:
    """Determine focal plane (z-index) from a 3D QP volume using a focus metric.
    Uses variance of Laplacian per slice and returns the argmax index.
    """
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
    #phi = phi_2d.astype(np.float64) - np.min(phi_2d)#phi_2d.astype(np.float64)
    phi = phi_2d.astype(np.float64)
    opd = phi*(wavelength_m / (2.0 * np.pi))# optical path difference in m (should be nm scale)
    area_element = pixel_size_m * pixel_size_m

    m = np.nansum(opd* area_element) / alpha_m3_per_g
    #m = (wavelength_m / (2.0 * np.pi * alpha_m3_per_kg)) * integral
    
    return float(m)

def dry_mass_per_cell(
    phi_fp: np.ndarray,
    mask: np.ndarray,
    wavelength_m: float,
    alpha_m3_per_g: float,
    pixel_size_m: float,
) -> np.ndarray:
    """Compute dry mass per cell and return a dict with visualisation, area and mass info per cell."""
    
    cell_masks = separate_masks(phi_fp, mask) 

    cell = {}
    pixel_area = pixel_size_m**2
    for ii, p in enumerate(cell_masks):
        print(f"Processing region {ii+1}/{len(cell_masks)}")
        #cell_data = p.image_intensity * pixel_area
        min_phase = np.min(p.image_intensity)
        cell_data = p.image_intensity - min_phase # correct for border effects
        area = p.area_filled * pixel_area  # area in m^2
        mass = dry_mass_from_qp(cell_data, wavelength_m, alpha_m3_per_g, pixel_size_m)
        cell[ii] = {}
        cell[ii]['mass'] = mass # np.nansum(cell_data) * wavelength_m / (2*np.pi*alpha) # mass in g 
        cell[ii]['area'] = area # area in m^2

        fig, ax = plt.subplots(dpi=100, constrained_layout=True)
        im = ax.imshow(cell_data, cmap='inferno', interpolation="nearest")
        ax.set_title(f"Region {ii+1} - Mass: {cell[ii]['mass']*10**(12):.4f} pg")
        ax.set_xticks([]); ax.set_yticks([])      # same as ax.axis('off') but keeps the frame if you want
        ax.set_frame_on(False)
            # Attach colorbar to THIS image
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
) -> np.ndarray:
    """Compute dry mass per cell and return a dict with visualisation, area and mass info per cell."""
    
    cell_masks = separate_masks(phi_fp, mask_cell) 
    cell_nuc = separate_masks(phi_fp, mask_nuc)

    cell = {}
    pixel_area = pixel_size_m**2
    for ii, p in enumerate(cell_nuc):
        print(f"Processing region {ii+1}/{len(cell_nuc)}")
        
        # find cell data that matches nucleus label
        label_map = {c.label: c for c in cell_masks}
        matching_cell = label_map.get(p.label)
        if matching_cell is None:
            min_phase = 0.0
        else: 
            min_phase = np.min(matching_cell.image_intensity)
        
        cell_data = p.image_intensity - min_phase
        area = p.area_filled * pixel_area  # area in m^2
        mass = dry_mass_from_qp(cell_data, wavelength_m, alpha_m3_per_g, pixel_size_m)
        cell[ii] = {}
        cell[ii]['mass'] = mass # np.nansum(cell_data) * wavelength_m / (2*np.pi*alpha) # mass in g 
        cell[ii]['area'] = area # area in m^2

        fig, ax = plt.subplots(dpi=100, constrained_layout=True)
        im = ax.imshow(cell_data, cmap='inferno', interpolation="nearest")
        ax.set_title(f"Region {ii+1} - Mass: {cell[ii]['mass']*10**(12):.4f} pg")
        ax.set_xticks([]); ax.set_yticks([])      # same as ax.axis('off') but keeps the frame if you want
        ax.set_frame_on(False)
            # Attach colorbar to THIS image
        cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.9)
        cbar.set_label("phase [rad]", rotation=270, labelpad=12)

        cell[ii]['figure'] = fig
        cell[ii]['data'] = cell_data

    
    return cell




def separate_masks(phi: np.ndarray, mask: np.ndarray, plot_flag = False) -> list:
    m=(mask>0).astype(np.int8)  # convert mask to int for multiplication
    img_masked = phi * m  # img # mask the image with the segmentation mask  
    img_masked[img_masked==0] = 0.000#None 

    if plot_flag:
        fig, ax = plt.subplots(1,2, figsize=(10,5))
        ax[0].imshow(phi, cmap='gray')
        ax[0].set_title('Original QP slice')
        ax[1].imshow(img_masked, cmap='gray')
        ax[1].set_title('Masked QP slice')
        plt.show()

    label = skim.label(mask)
    props = skim.regionprops(label, intensity_image=img_masked)
    return props

# ------------------------- High-level convenience wrapper ------------------------

@dataclass
class PipelineConfig:
    channel_bf: int = 0              # index into c-axis for BF channel (when input is 4D)
    seg_reduce: str = "focal"         # how to collapse channels for segmentation: 'mean' or 'max'
    opening_radius: int = 1
    min_size: int = 64
    wavelength_m: float = 550e-9     # default 550 nm
    alpha_m3_per_g: float = 0.18e-3 # typical for proteins (mL/g -> m^3/kg); adjust for your sample
    pixel_size_m: float = 0.1e-6     # 100 nm pixels; adjust per system
    ps: phase_structure = phase_structure()  # TPR structure parameters
    fp: int = -1                      # focal plane index for segmentation when method='focal'
    model_type: str = "cyto"          # Cellpose model type for segmentation
    flow_threshold: float = 0.4
    cell_threshold: float = 0.0
    overwrite_qp: bool = False      # whether to overwrite existing QP results
    cellpose_model = None          # pre-loaded Cellpose model (optional)
    invert_stack: bool = False    # whether to invert the stack for QP retrieval
    segment_nucleus : bool = False  # whether to segment nucleus 
    channel_nuc: int = 1          # channel index for nucleus segmentation (if enabled)
    cell_area_threshold: int = 6400  # pixel wise area for cell segmentation: ca 90 pixel diameter circle

@dataclass
class PipelineOutputs:
    phi_3d: np.ndarray           # (z,y,x) QP volume from BF
    seg_mask: np.ndarray      # (z,y,x) boolean mask
    z_focal: int                 # best focal plane index
    dry_masses: dict     # dry mass at the focal plane (kg)

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
        # 1) QP retrieval (3D) regular way 
        phi_3d, fl_channels = qp_retrieve_3d_from_channel(
            data=data, channel_bf=cfg.channel_bf,
            phase_str=cfg.ps,
            compute_qp=True,
            invert_stack=cfg.invert_stack
        )
        write_qp_stack_to_tiff(phi_3d, path_save, file_out, cfg.ps)
        write_phase_structure(cfg.ps, path_save, filename)
        phi_3d = load_data(filepath)
    else:
        # 1) split incoming stack and load QP from cached file
        _, fl_channels = qp_retrieve_3d_from_channel(
            data=data, channel_bf=cfg.channel_bf, 
            phase_str=cfg.ps, 
            compute_qp=False, 
            invert_stack=cfg.invert_stack
        )
        phi_3d = load_data(filepath)


    # 2) Focal plane from QP
    if cfg.fp >= 0:
        z_focal = cfg.fp  # override if specified in config
    else:
        z_focal = determine_focal_plane_from_qp(phi_3d)
    
    
 
    # 3) Segmentation using all available data
    if cfg.cellpose_model is None:
        model = setup_cellpose_model(model_type=cfg.model_type) 

    seg_mask, phi_fp, flow, fig= segment_volume(
        data=phi_3d, method=cfg.seg_reduce, 
        model_type=cfg.model_type,
        model=model,
        fp=z_focal, 
        fluorescence_channels=fl_channels, 
        flow_threshold=cfg.flow_threshold,
        cell_threshold=cfg.cell_threshold
    )

    seg_mask = filter_masks_by_size(seg_mask, min_size=cfg.cell_area_threshold)
    fig = plot_segmentation_results(phi_fp, seg_mask, flow)
    write_segmentation(fig, path_save, file_id)

    # 3.2) Optionally segment nucleus only
    if cfg.segment_nucleus and fl_channels is not None:
        nuc_mask, nuc_img, nuc_fig = segment_nucleus(
            data=np.squeeze(fl_channels[cfg.channel_nuc-1,:,:,:]),
            method=cfg.seg_reduce,
            fp=z_focal
        )

        nuc_mask = link_nucleus_to_cells(nuc_mask, seg_mask)
        write_segmentation(nuc_fig, path_save, file_id+'_nucleus')

    # 4) Dry mass conversion (per z and at focal)
    dry_masses = dry_mass_per_cell(
        phi_fp=phi_fp,
        mask=seg_mask,
        wavelength_m=cfg.wavelength_m,
        alpha_m3_per_g=cfg.alpha_m3_per_g,
        pixel_size_m=cfg.pixel_size_m,
    )   

    # 5) write output results
    write_cell_images(dry_masses, path_save, file_id if file_id is not None else 'output')
    write_mask(img=phi_fp, mask=seg_mask, flow=flow, path_save=path_save, filename=file_id)
    write_dataframe(dry_masses, path_save, file_id)

    # 4.2) Optionally compute dry mass for nucleus
    if cfg.segment_nucleus and fl_channels is not None:
        dry_masses = dry_mass_nuc_per_cell(
            phi_fp=phi_fp,
            mask_cell=seg_mask,
            mask_nuc=nuc_mask,
            wavelength_m=cfg.wavelength_m,
            alpha_m3_per_g=cfg.alpha_m3_per_g,
            pixel_size_m=cfg.pixel_size_m,
        )   
        write_cell_images(dry_masses, path_save, file_id+'_nucleus')
        write_mask(img=phi_fp, mask=nuc_mask, flow=flow, path_save=path_save, filename=file_id+'_nucleus')
        write_dataframe(dry_masses, path_save, file_id+'_nucleus')



    return PipelineOutputs(
        phi_3d=phi_3d,
        seg_mask=seg_mask,
        z_focal=z_focal,
        dry_masses=dry_masses
    )


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


# ------------------------- nucleus helpers -----------------------

def get_nucleus_mask_from_image(
    image: np.ndarray, area_threshold: int = 100, nuc_distance: int = 25):
    # nucleus smooth nucleus images
    image = ski.filters.gaussian(image, sigma=3) #12
    #thresh = ski.filters.threshold_otsu(image)
    thresh = ski.filters.threshold_yen(image)
    binary = image > thresh

    distance = ndi.distance_transform_edt(binary)
    coords = ski.feature.peak_local_max(distance, footprint=np.ones((7, 7)), min_distance=nuc_distance, labels=binary)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = ski.segmentation.watershed(-distance, markers, mask=binary)
    labels = ski.segmentation.expand_labels(labels, distance=1)

    label_props = ski.measure.regionprops(labels) 
    # clean up masks with size threshold
    for l in label_props:
        if l.area < area_threshold:
            labels[labels==l.label] = 0

    return labels

def link_nucleus_to_cells(nuc_mask: np.ndarray, cell_mask: np.ndarray) -> np.ndarray:

    nmask_props = ski.measure.regionprops(nuc_mask)
    cmask_props = ski.measure.regionprops(cell_mask)
    nmask_new  = np.zeros_like(nuc_mask)                          
    # for every nucleus in the mask look whether it overlaps with a cell in the cell mask and adjust the label of it
    for ii, nprop in enumerate(nmask_props):

        # check whether nucleus centroid falls into label area in cell masks and write nucleus label accordingly
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
    if data.shape[0] != data.shape[1] & data.shape[1] == data.shape[2]:
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
    """Write segmentation mask to a TIFF file."""
    if not os.path.exists(path_save):
        os.mkdir(path_save)
        print("Created %s " % path_save)
    
    io.save_masks(img, mask, flow, file_names=os.path.join(path_save, filename))



def write_dataframe(cell: dict, path_save: str, filename: str):
    """Write dry mass data to a CSV file."""
    if not os.path.exists(path_save):
        os.mkdir(path_save)
        print("Created %s " % path_save)

    if any(cell.items()):# check if dict is not empty
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
    #fig.close()