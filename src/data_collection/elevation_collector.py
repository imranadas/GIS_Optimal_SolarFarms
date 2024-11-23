# src/data_collection/elevation_collector.py

import ee
import logging
import requests
import rasterio
import numpy as np
from pathlib import Path
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling

class ElevationCollector:
    """Collects elevation data from various sources including SRTM, ASTER, and Google Earth Engine."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(config.RAW_DATA_DIR) / 'elevation'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Earth Engine
        try:
            ee.Initialize()
        except Exception as e:
            self.logger.warning(f"Earth Engine initialization failed: {e}")
    
    def get_srtm_gee(self):
        """
        Fetch SRTM elevation data from Google Earth Engine.
        Resolution: 30m
        """
        try:
            # Define region of interest
            roi = ee.Geometry.Rectangle(
                self.config.ROI_BOUNDS['west'],
                self.config.ROI_BOUNDS['south'],
                self.config.ROI_BOUNDS['east'],
                self.config.ROI_BOUNDS['north']
            )
            
            # Get SRTM dataset
            srtm = ee.Image('USGS/SRTMGL1_003').clip(roi)
            
            # Export to GeoTIFF
            output_path = self.output_dir / 'srtm_gee.tif'
            
            # Get the data
            url = srtm.getDownloadURL({
                'scale': 30,
                'crs': 'EPSG:4326',
                'region': roi,
                'format': 'GEO_TIFF'
            })
            
            # Download the file
            response = requests.get(url)
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            self.logger.info(f"SRTM data downloaded to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error fetching SRTM data from GEE: {e}")
            return None
    
    def get_aster_gdem(self):
        """
        Fetch ASTER GDEM data from NASA Earth Data.
        Resolution: 30m
        Note: Requires Earth Data login credentials in config
        """
        try:
            # Build URL for the region
            base_url = "https://e4ftl01.cr.usgs.gov/ASTT/ASTGTM.003/2000.03.01/"
            
            # Calculate tile numbers based on lat/lon
            lat_north = int(self.config.ROI_BOUNDS['north'])
            lat_south = int(self.config.ROI_BOUNDS['south'])
            lon_west = int(self.config.ROI_BOUNDS['west'])
            lon_east = int(self.config.ROI_BOUNDS['east'])
            
            output_files = []
            
            # Download each tile that covers the area
            for lat in range(lat_south, lat_north + 1):
                for lon in range(lon_west, lon_east + 1):
                    # Format tile name
                    ns = 'N' if lat >= 0 else 'S'
                    ew = 'E' if lon >= 0 else 'W'
                    lat_str = f"{abs(lat):02d}"
                    lon_str = f"{abs(lon):03d}"
                    tile = f"ASTGTM_NC_{ns}{lat_str}{ew}{lon_str}"
                    
                    url = f"{base_url}{tile}_dem.tif"
                    output_path = self.output_dir / f"{tile}.tif"
                    
                    # Download tile
                    if not output_path.exists():
                        response = requests.get(url, auth=(self.config.NASA_USERNAME, self.config.NASA_PASSWORD))
                        if response.status_code == 200:
                            with open(output_path, 'wb') as f:
                                f.write(response.content)
                            output_files.append(output_path)
                            
            return output_files
            
        except Exception as e:
            self.logger.error(f"Error fetching ASTER GDEM data: {e}")
            return None
    
    def merge_elevation_tiles(self, tile_paths, output_name='merged_elevation.tif'):
        """Merge multiple elevation tiles into a single file"""
        try:
            # Open all raster files
            src_files = [rasterio.open(p) for p in tile_paths]
            
            # Merge tiles
            merged, transform = merge(src_files)
            
            # Get metadata from first file
            meta = src_files[0].meta.copy()
            meta.update({
                'height': merged.shape[1],
                'width': merged.shape[2],
                'transform': transform
            })
            
            # Write merged file
            output_path = self.output_dir / output_name
            with rasterio.open(output_path, 'w', **meta) as dest:
                dest.write(merged)
            
            # Close source files
            for src in src_files:
                src.close()
                
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error merging elevation tiles: {e}")
            return None
    
    def process_elevation_data(self, input_path, output_name='processed_elevation.tif'):
        """Process elevation data: reproject, clip to ROI, and fill voids"""
        try:
            with rasterio.open(input_path) as src:
                # Calculate transform for target CRS (WGS84)
                transform, width, height = calculate_default_transform(
                    src.crs, 'EPSG:4326',
                    src.width, src.height,
                    *src.bounds
                )
                
                # Update metadata
                meta = src.meta.copy()
                meta.update({
                    'crs': 'EPSG:4326',
                    'transform': transform,
                    'width': width,
                    'height': height
                })
                
                # Create output file
                output_path = self.output_dir / output_name
                with rasterio.open(output_path, 'w', **meta) as dest:
                    # Reproject and write data
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=rasterio.band(dest, 1),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs='EPSG:4326',
                        resampling=Resampling.bilinear
                    )
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error processing elevation data: {e}")
            return None
    
    def get_elevation_data(self, source='srtm', process=True):
        """
        Main method to get elevation data from specified source
        
        Args:
            source (str): 'srtm' or 'aster'
            process (bool): Whether to process the data after downloading
        """
        try:
            if source.lower() == 'srtm':
                elevation_path = self.get_srtm_gee()
            elif source.lower() == 'aster':
                tile_paths = self.get_aster_gdem()
                elevation_path = self.merge_elevation_tiles(tile_paths)
            else:
                raise ValueError(f"Unsupported elevation source: {source}")
            
            if process and elevation_path:
                elevation_path = self.process_elevation_data(elevation_path)
            
            return elevation_path
            
        except Exception as e:
            self.logger.error(f"Error in get_elevation_data: {e}")
            return None

    def fill_voids(self, dem_path):
        """Fill voids in DEM using interpolation"""
        try:
            with rasterio.open(dem_path) as src:
                dem = src.read(1)
                meta = src.meta.copy()
                
                # Identify voids (usually marked as -32768 or similar)
                voids = (dem < -1000) | (dem > 9000) | np.isnan(dem)
                
                if not np.any(voids):
                    return dem_path
                
                # Simple interpolation for void filling
                from scipy.ndimage import gaussian_filter
                
                # Create a mask of valid values
                valid = ~voids
                
                # Create output array
                filled = dem.copy()
                
                # Fill voids with gaussian filter
                filled[voids] = gaussian_filter(
                    dem, sigma=2,
                    mode='constant',
                    cval=dem[valid].mean()
                )[voids]
                
                # Save filled DEM
                output_path = str(dem_path).replace('.tif', '_filled.tif')
                with rasterio.open(output_path, 'w', **meta) as dest:
                    dest.write(filled, 1)
                
                return output_path
                
        except Exception as e:
            self.logger.error(f"Error filling voids in DEM: {e}")
            return None
