# src/data_collection/osm_collector.py

import osmnx as ox
from shapely.geometry import box

class OSMDataCollector:
    def __init__(self, config):
        self.config = config
        self.bbox = box(
            self.config.ROI_BOUNDS['west'],
            self.config.ROI_BOUNDS['south'],
            self.config.ROI_BOUNDS['east'],
            self.config.ROI_BOUNDS['north']
        )
        
    def get_road_network(self):
        """Collect road network data from OSM"""
        try:
            roads = ox.graph_from_bbox(
                self.config.ROI_BOUNDS['north'],
                self.config.ROI_BOUNDS['south'],
                self.config.ROI_BOUNDS['east'],
                self.config.ROI_BOUNDS['west'],
                network_type='drive'
            )
            
            # Convert to GeoDataFrame
            roads_gdf = ox.graph_to_gdfs(roads)[1]
            
            # Save to file
            roads_gdf.to_file(
                f"{self.config.RAW_DATA_DIR}/roads.gpkg",
                driver='GPKG'
            )
            return roads_gdf
            
        except Exception as e:
            print(f"Error collecting road network data: {e}")
            return None
    
    def get_power_infrastructure(self):
        """Collect power infrastructure data from OSM"""
        try:
            tags = {'power': ['line', 'minor_line', 'substation']}
            power_infra = ox.geometries_from_bbox(
                self.config.ROI_BOUNDS['north'],
                self.config.ROI_BOUNDS['south'],
                self.config.ROI_BOUNDS['east'],
                self.config.ROI_BOUNDS['west'],
                tags
            )
            
            # Save to file
            power_infra.to_file(
                f"{self.config.RAW_DATA_DIR}/power_infrastructure.gpkg",
                driver='GPKG'
            )
            return power_infra
            
        except Exception as e:
            print(f"Error collecting power infrastructure data: {e}")
            return None
    
    def get_residential_areas(self):
        """Collect residential area data from OSM"""
        try:
            tags = {'landuse': ['residential'], 'building': True}
            residential = ox.geometries_from_bbox(
                self.config.ROI_BOUNDS['north'],
                self.config.ROI_BOUNDS['south'],
                self.config.ROI_BOUNDS['east'],
                self.config.ROI_BOUNDS['west'],
                tags
            )
            
            # Save to file
            residential.to_file(
                f"{self.config.RAW_DATA_DIR}/residential_areas.gpkg",
                driver='GPKG'
            )
            return residential
            
        except Exception as e:
            print(f"Error collecting residential area data: {e}")
            return None
