import nibabel as nib
import numpy as np
from dipy.viz import window, actor
from fury.actor import slicer
from fury.colormap import colormap_lookup_table
import vtk
import pdb 
from pathlib import Path
import matplotlib.pyplot as plt 

def plot_nifti(
    nifti_path,
    data_slice="m",
    orientation="axial",
    size=(600, 400),
    value_range=None,
    save_path=None,
    interactive=True,
    scalar_colorbar=True,
    tractography=None,
    tractography_values=None,
    tractography_cmap=None,
    tractography_cmap_range=None,
    tractography_colorbar=False,
    volume_idx=None,
    visualize_tensor=False,
    visualize_odf=False,
    sh_order_max=6,
    sh_basis='real',
    **kwargs,
):
    """Create a 2D rendering of a NIFTI slice.

    Parameters
    ----------
    nifti_path : str or Path
        Path to NIFTI to plot
    data_slice : str or int, optional
        Slice to plot or "m" for middle slice, by default "m"
    orientation : str, optional
        Can be "axial", "sagittal" or "coronal", by default "axial"
    size : tuple, optional
        Size of window, by default (600,400)
    save_path : str or Path, optional
        Optional path to save to, by default None
    interactive : bool, optional
        Whether to interactively show the scene, by default True
    scalar_colorbar : bool, optional
        Whether to show a scalar colorbar (for FA, T1, etc.), by default True
    tractography : list of str or Path, optional
        Optional tractogram(s) to plot with slices. Can provide multiple files, by default None
    tractography_values : list of float, optional
        Optional values to color the tractography with, by default None
    tractography_cmap : str, optional
        Optional colormap to use for the tractography, by default None
    tractography_cmap_range : list of float, optional
        Optional range to use for the colormap, by default None for 0 to 1
    tractography_colorbar : bool, optional
        Whether to show a colorbar for the tractography, by default False
    volume_idx : int, optional
        Index of the volume to display if the image is 4D, by default None
    **kwargs
        Additional keyword arguments to pass to fury.actor.slicer
    """

    # Load the data and convert to RAS
    nifti = nib.load(nifti_path)
    nifti = nib.as_closest_canonical(nifti)
    data = nifti.get_fdata()

    
    if len(data.shape) == 4:
        if volume_idx is None:
            raise ValueError(
                "Input is a 4D image but no volume index specified. "
                "Please provide a volume_idx parameter to select which 3D volume to display."
            )
        if not 0 <= volume_idx < data.shape[3]:
            raise ValueError(
                f"volume_idx {volume_idx} is out of bounds for image with {data.shape[3]} volumes"
            )
        data = data[..., volume_idx]
    elif len(data.shape) != 3:
        raise ValueError(
            f"Expected 3D or 4D image, but got image with {len(data.shape)} dimensions"
        )

    # Get the data and affine
    affine = nifti.affine

    # value range
    if value_range is None:
        value_range = [np.min(data), np.max(data)]
    else:
        value_range = [value_range[0], value_range[1]]

    # Set up slicer and window
    slice_actor = slicer(data, affine=affine, value_range=value_range, **kwargs)
    scene = window.Scene()
    scene.add(slice_actor)

    # Get slice if not defined
    if orientation == "axial":
        data_slice = data.shape[2] // 2 if data_slice == "m" else data_slice
        slice_actor.display_extent(
            0, data.shape[0], 0, data.shape[1], data_slice, data_slice
        )

        camera_pos = (0, 0, 1)
        camera_up = (0, 1, 0)
    elif orientation == "coronal":
        data_slice = data.shape[1] // 2 if data_slice == "m" else data_slice
        slice_actor.display_extent(
            0, data.shape[0], data_slice, data_slice, 0, data.shape[2]
        )

        camera_pos = (0, 1, 0)
        camera_up = (0, 0, 1)
    elif orientation == "sagittal":
        data_slice = data.shape[0] // 2 if data_slice == "m" else data_slice
        slice_actor.display_extent(
            data_slice, data_slice, 0, data.shape[1], 0, data.shape[2]
        )

        camera_pos = (1, 0, 0)
        camera_up = (0, 0, 1)
    camera_focal = (0, 0, 0)

    # Apply colorbar
    if scalar_colorbar:
        # Create a grayscale colormap (from black to white)
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)  # Full grayscale (256 levels)
        lut.Build()  # Initialize the LUT
        
        
        for i in range(256):
            lut.SetTableValue(i, i / 255.0, i / 255.0, i / 255.0, 1)  # Grayscale colors
        '''
        We can further optimize this later to support orther colormaps; this just supports grayscale right now.
        '''
        # Set the full grayscale range (e.g., 0 to 255 for typical image data)
        lut.SetRange(value_range[0], value_range[1])  # This defines the grayscale range explicitly

        # Create the scalar bar (colorbar)
        scalar_bar = vtk.vtkScalarBarActor()
        scalar_bar.SetLookupTable(lut)  # Attach the grayscale LUT
        scalar_bar.SetLabelFormat("%.0f")  # Integer labels (0, 1, 2, ... 255)
        scalar_bar.SetPosition(0.8, 0.1)  # Position of the colorbar
        scalar_bar.SetHeight(0.5)  # Adjust height (increase size)
        scalar_bar.SetWidth(0.1)   # Adjust width (increase size)

        # Add the scalar bar to the scene
        scene.add(scalar_bar)
    # Add tractography
    if tractography is not None:
        if tractography_cmap is None:
            tractography_cmap = "Set1" if tractography_values is None else "plasma"
        cmap = plt.get_cmap(tractography_cmap)

        if tractography_cmap_range is None:
            tractography_cmap_range = [0, 1]

        # Set to range
        if tractography_values is not None:
            norm = plt.Normalize(vmin=tractography_cmap_range[0], vmax=tractography_cmap_range[1])
            colors = [cmap(norm(val)) for val in tractography_values]
        else:
            colors = [cmap(i) for i in range(len(tractography))]

        # Apply colorbar
        if tractography_colorbar:
            # Create a grayscale colormap (from black to white)
            lut = vtk.vtkLookupTable()
            lut.SetNumberOfTableValues(256)  # Full grayscale (256 levels)
            lut.Build()  # Initialize the LUT
            
            cmap_range = np.linspace(tractography_cmap_range[0], tractography_cmap_range[1], 256)
            for i in range(256):
                # Sample the colormap
                color = cmap(norm(cmap_range[i]))
                lut.SetTableValue(i, color[0], color[1], color[2], 1)  # Grayscale colors

            if tractography_cmap_range is not None:
                lut.SetRange(tractography_cmap_range[0], tractography_cmap_range[1])
            else:
                lut.SetRange(0, 1)

            # Create the scalar bar (colorbar)
            tract_bar = vtk.vtkScalarBarActor()
            tract_bar.SetLookupTable(lut)  # Attach the grayscale LUT
            tract_bar.SetLabelFormat("%.2f")  # Labels
            tract_bar.SetPosition(0.1, 0.1)  # Position of the colorbar
            tract_bar.SetHeight(0.5)  # Adjust height (increase size)
            tract_bar.SetWidth(0.1)   # Adjust width (increase size)

            # Add the scalar bar to the scene
            scene.add(tract_bar)
            
    # Add each tractography with its corresponding color
    for tract_file, color in zip(tractography, colors):
        streamlines = nib.streamlines.load(tract_file).streamlines
        stream_actor = actor.line(streamlines, colors=color, linewidth=0.2)
        scene.add(stream_actor)

    # Set up camera
    scene.set_camera(position=camera_pos, focal_point=camera_focal, view_up=camera_up)

    # Show the scene
    if save_path:
        window.record(scene=scene, out_path=save_path, size=size, reset_camera=True)

    if interactive:
        window.show(scene, size=size, reset_camera=True)




