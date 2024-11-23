# src/visualization/map_visualizer.py

import folium
import logging
import numpy as np
from pathlib import Path
import branca.colormap as cm
from typing import Optional, Dict

class MapVisualizer:
    """Creates interactive visualizations of solar farm suitability analysis results"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(config.PROCESSED_DATA_DIR)
        
        # Define color schemes
        self.suitability_colors = {
            'continuous': ['#d73027', '#fc8d59', '#fee090', '#e0f3f8', '#91bfdb', '#4575b4'],
            'classified': {
                4: '#1a9850',  # Most suitable
                3: '#91cf60',  # Suitable
                2: '#d9ef8b',  # Moderately suitable
                1: '#fee08b',  # Unsuitable
                0: '#d73027'   # Restricted
            }
        }
        
    def create_suitability_map(
        self,
        suitability_data: np.ndarray,
        classified_data: np.ndarray,
        additional_layers: Optional[Dict] = None
    ) -> folium.Map:
        """
        Create interactive map with suitability results
        
        Args:
            suitability_data: Continuous suitability scores
            classified_data: Classified suitability
            additional_layers: Optional additional layers to display
            
        Returns:
            folium.Map: Interactive map
        """
        try:
            # Initialize map centered on ROI
            center_lat = (self.config.ROI_BOUNDS['north'] + 
                        self.config.ROI_BOUNDS['south']) / 2
            center_lon = (self.config.ROI_BOUNDS['east'] + 
                        self.config.ROI_BOUNDS['west']) / 2
            
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=10,
                tiles='cartodbpositron'
            )
            
            # Add base layer control
            folium.LayerControl().add_to(m)
            
            # Add continuous suitability layer
            continuous_colormap = cm.LinearColormap(
                colors=self.suitability_colors['continuous'],
                vmin=0,
                vmax=1,
                caption='Suitability Score'
            )
            
            self._add_raster_layer(
                m,
                suitability_data,
                'Suitability (Continuous)',
                continuous_colormap,
                opacity=0.7
            )
            
            # Add classified suitability layer
            classified_colormap = cm.LinearColormap(
                colors=list(self.suitability_colors['classified'].values()),
                vmin=0,
                vmax=4,
                caption='Suitability Class'
            )
            
            self._add_raster_layer(
                m,
                classified_data,
                'Suitability (Classified)',
                classified_colormap,
                opacity=0.7
            )
            
            # Add additional layers if provided
            if additional_layers:
                for name, layer in additional_layers.items():
                    self._add_additional_layer(m, layer, name)
            
            # Add legend
            self._add_legend(m)
            
            # Add scale bar
            folium.plugins.MeasureControl().add_to(m)
            
            return m
            
        except Exception as e:
            self.logger.error(f"Error creating suitability map: {e}")
            return None

    def _add_raster_layer(
        self,
        map_obj: folium.Map,
        data: np.ndarray,
        name: str,
        colormap: cm.LinearColormap,
        opacity: float = 0.7
    ):
        """Add a raster layer to the map"""
        try:
            # Convert numpy array to GeoJSON
            geojson = self._array_to_geojson(
                data,
                self.config.ROI_BOUNDS
            )
            
            # Add GeoJSON layer
            folium.GeoJson(
                geojson,
                name=name,
                style_function=lambda x: {
                    'fillColor': colormap(x['properties']['value']),
                    'color': 'none',
                    'fillOpacity': opacity
                }
            ).add_to(map_obj)
            
        except Exception as e:
            self.logger.error(f"Error adding raster layer: {e}")

    def _add_additional_layer(
        self,
        map_obj: folium.Map,
        layer_data: Dict,
        name: str
    ):
        """Add additional vector or raster layers to the map"""
        try:
            if layer_data['type'] == 'vector':
                folium.GeoJson(
                    layer_data['data'],
                    name=name,
                    style_function=layer_data.get('style_function', None)
                ).add_to(map_obj)
            elif layer_data['type'] == 'raster':
                self._add_raster_layer(
                    map_obj,
                    layer_data['data'],
                    name,
                    layer_data['colormap'],
                    layer_data.get('opacity', 0.7)
                )
                
        except Exception as e:
            self.logger.error(f"Error adding additional layer: {e}")

    def _add_legend(self, map_obj: folium.Map):
        """Add legend to the map"""
        legend_html = """
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; 
                    border:2px solid grey; z-index:9999; 
                    background-color:white;
                    padding: 10px;
                    font-size:14px;
                    font-family: Arial, sans-serif;">
        <p><strong>Suitability Classes</strong></p>
        """
        
        for class_value, color in self.suitability_colors['classified'].items():
            label = {
                4: 'Most Suitable',
                3: 'Suitable',
                2: 'Moderately Suitable',
                1: 'Unsuitable',
                0: 'Restricted'
            }[class_value]
            
            legend_html += f"""
            <p><i class="fa fa-square fa-1x"
                 style="color:{color}"></i> {label}</p>
            """
        
        legend_html += "</div>"
        map_obj.get_root().html.add_child(folium.Element(legend_html))

    def _array_to_geojson(
        self,
        array: np.ndarray,
        bounds: Dict[str, float]
    ) -> Dict:
        """Convert numpy array to GeoJSON format"""
        features = []
        rows, cols = array.shape
        
        lat_step = (bounds['north'] - bounds['south']) / rows
        lon_step = (bounds['east'] - bounds['west']) / cols
        
        for i in range(rows):
            for j in range(cols):
                if not np.isnan(array[i, j]):
                    features.append({
                        'type': 'Feature',
                        'geometry': {
                            'type': 'Polygon',
                            'coordinates': [[
                                [bounds['west'] + j * lon_step, 
                                 bounds['north'] - i * lat_step],
                                [bounds['west'] + (j+1) * lon_step, 
                                 bounds['north'] - i * lat_step],
                                [bounds['west'] + (j+1) * lon_step, 
                                 bounds['north'] - (i+1) * lat_step],
                                [bounds['west'] + j * lon_step, 
                                 bounds['north'] - (i+1) * lat_step],
                                [bounds['west'] + j * lon_step, 
                                 bounds['north'] - i * lat_step]
                            ]]
                        },
                        'properties': {
                            'value': float(array[i, j])
                        }
                    })
        
        return {
            'type': 'FeatureCollection',
            'features': features
        }

    def save_map(self, map_obj: folium.Map, filename: str):
        """Save map to HTML file"""
        try:
            output_path = self.output_dir / filename
            map_obj.save(str(output_path))
            self.logger.info(f"Map saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving map: {e}")
