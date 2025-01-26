import argparse
from pathlib import Path

from open_dive.viz import plot_nifti


def main():
    # Create args
    parser = argparse.ArgumentParser(description="Plot a slice of a NIFTI file")
    parser.add_argument("nifti_path", type=Path, help="Path to NIFTI to plot")
    parser.add_argument(
        "-s", "--slice", default="m", help='Slice to plot or "m" for middle slice'
    )
    parser.add_argument(
        "-o",
        "--orientation",
        default="axial",
        help='Can be "axial", "sagittal" or "coronal"',
    )
    parser.add_argument(
        "--size", default=(600, 400), help="Size of window, by default (600,400)"
    )
    parser.add_argument("--save_path", help="Optional path to save to")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Whether to interactively show the scene",
    )
    parser.add_argument(
        "--value_range",
        type=int,
        nargs=2,
        help="Optional value range to pass to slicer. Default is min/max of image.",
    )
    parser.add_argument(
        "--volume_idx",
        type=int,
        help="Index of the volume to display if the image is 4D",
    )
    parser.add_argument(
        "--interpolation",
        default="nearest",
        help="Interpolation method to use (nearest or linear). Default is 'nearest'.",
    )
    parser.add_argument(
        "--scalar_colorbar",
        action="store_true",
        help="Whether to show a colorbar, by default True",
    )
    parser.add_argument(
        ## plot tractogram with slices
        "--tractography",
        type=Path,
        nargs="+",  # Accept one or more arguments
        help="Optional tractogram(s) to plot with slices. Can provide multiple files.",
    )

    
 # Add tensor visualization option
    parser.add_argument(
        "--visualize_tensor",
        action="store_true",
        help="Whether to visualize diffusion tensors on the NIFTI image",
    )
    
    # Add ODF visualization option
    parser.add_argument(
        "--visualize_odf",
        action="store_true",
        help="Whether to visualize ODFs on the NIFTI image",
    )
    
    # Add spherical harmonic max order for ODF
    parser.add_argument(
        "--sh_order_max",
        type=int,
        default=6,
        help="Maximum spherical harmonic order for ODF visualization (default 6)",
    )
    
    # Add basis for ODF spherical harmonics
    parser.add_argument(
        "--sh_basis",
        choices=["real", "complex"],
        default="real",
        help="Basis for spherical harmonics: 'real' or 'complex' (default 'real')",
    )

 

    args = parser.parse_args()

    # Plot the NIFTI
    plot_nifti(
        args.nifti_path,
        data_slice=args.slice,
        orientation=args.orientation,
        size=args.size,
        volume_idx=args.volume_idx,
        save_path=args.save_path,
        interactive=args.interactive,
        value_range=args.value_range,
        interpolation=args.interpolation,
        scalar_colorbar=args.scalar_colorbar,
        tractography=args.tractography,
        visualize_tensor=args.visualize_tensor,  # Pass the new argument
        visualize_odf=args.visualize_odf,        # Pass the new argument
        sh_order_max=args.sh_order_max,          # Pass the ODF spherical harmonic order
        sh_basis=args.sh_basis, 

    )
if __name__ == "__main__":
    main()

