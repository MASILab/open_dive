from matplotlib.colors import LinearSegmentedColormap
import numpy as np

class TractVisualizer:
    def _create_thickness_panel(self, ax):
        """Create the upper panel showing thickness"""
        ax.set_facecolor('black')
        
        # Create custom colormap for thickness
        colors = ['black', 'teal', 'orange', 'white']
        thickness_cmap = LinearSegmentedColormap.from_list('thickness', colors)
        
        for tract, value in zip(self.tracts, self.tract_values):
            streamlines = tract.streamlines
            if value is not None:
                # Normalize value to colormap range
                norm_value = (value - self.thickness_range[0]) / (self.thickness_range[1] - self.thickness_range[0])
                color = thickness_cmap(norm_value)
            else:
                color = 'white'  # Default color if no value provided
                
            # Plot each streamline
            for streamline in streamlines:
                points = np.array(streamline)
                ax.plot(points[:, 0], points[:, 1], color=color, alpha=0.5, linewidth=0.5)
        
        ax.set_title('Cortical Thickness (mm)', color='white', pad=20)
        ax.axis('off')
        
    def _create_change_panel(self, ax):
        """Create the lower panel showing change"""
        ax.set_facecolor('black')
        
        # Create custom colormap for change
        colors = ['blue', 'white', 'red']
        change_cmap = LinearSegmentedColormap.from_list('change', colors)
        
        for tract, value in zip(self.tracts, self.tract_values):
            streamlines = tract.streamlines
            if value is not None:
                # Normalize value to colormap range
                norm_value = (value - self.change_range[0]) / (self.change_range[1] - self.change_range[0])
                color = change_cmap(norm_value)
            else:
                color = 'white'  # Default color if no value provided
                
            # Plot each streamline
            for streamline in streamlines:
                points = np.array(streamline)
                ax.plot(points[:, 0], points[:, 1], color=color, alpha=0.5, linewidth=0.5)
        
        ax.set_title('Cortical Thickness Change (%/year)', color='white', pad=20)
        ax.axis('off') 