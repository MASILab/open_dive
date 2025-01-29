# OpenDIVE
OpenDIVE (Open Diffusion Imaging Visualization for Everyone) is a command line interface tool for generating accessible, interpretable visualizations from diffusion MRI, initiated at [BrainHack Vandy 2025](https://brainhack-vandy.github.io/).

Despite the prolific availability of software tools to visualize diffusion MRI data, there is no standardized visualization to summarize longitudinal changes. Similarly, the current standard of visualizations are not accessible to people with common forms of colorblindness. We propose a software package to both standardize and improve accessibility to representations of diffusion data. 


Aim 1: [We propose a colorblind-friendly colormap for diffusion MRI images.](https://neurolabusc.github.io/OpenDIVE/)

Aim 2: We generate a standardized display for anatomical images to overlay the diffusion models based on user input, including support for multiple track files, color maps, and illustration of bundle summary metrics (p-value, volume, effect size, etc.) per bundle.

For FAQs related to diffusion MRI, see our diffusion FAQ issue (#4).

## Installation

You can install the package using Python:

```bash
pip install git+https://github.com/MASILab/open_dive
```

## Usage

After installing the package, you should be able to use the `nifti2png` command to produce images. To view your visualizations, use the `--interactive` flag. Please see the [wiki](https://github.com/MASILab/open_dive/wiki) for documentation on using the command.

## Contributing

We welcome issues and pull requests! For details on contributing, please see [CONTRIBUTING.md](CONTRIBUTING.md).

## Contributors
- [Adam Saunders](https://github.com/saundersresearch)
- [Elyssa McMaster](https://github.com/ElyssaMcMaster)
- [Chris Rorden](https://github.com/neurolabusc/neurolabusc)
- [Johaan Kathilankal Jis](https://github.com/johaankjis)
- [Minyi Sun](https://github.com/Orekiwlg)
- [Adam Sadriddinov](https://github.com/mukhsadr)
- [Lukas VanTilburg](https://github.com/beeper-weepers)
- [Kurt Schilling](https://github.com/schillkg)

## License

[MIT](https://choosealicense.com/licenses/mit/)
