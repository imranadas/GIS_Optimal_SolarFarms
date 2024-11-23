# src/data_collection/gee_collector.py

import ee
import geemap
import logging
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class GEEData:
    """Data class to store GEE collection results"""
    elevation: ee.Image
    solar_radiation: ee.Image
    land_cover: ee.Image
    file_paths: Dict[str, Path]

class GEEDataCollector:
    """Collects and processes Earth Engine data for solar farm site analysis"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        try:
            ee.Initialize()
            self.roi = ee.Geometry.Rectangle(
                self.config.ROI_BOUNDS['west'],
                self.config.ROI_BOUNDS['south'], 
                self.config.ROI_BOUNDS['east'],
                self.config.ROI_BOUNDS['north']
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize Earth Engine: {e}")
            raise

    def get_elevation_data(self) -> Optional[ee.Image]:
        """
        Collect and process SRTM elevation data.
        
        Returns:
            ee.Image: Processed elevation image or None if failed
        """
        try:
            srtm = ee.Image('USGS/SRTMGL1_003')
            elevation = (srtm
                .clip(self.roi)
                .setDefaultProjection('EPSG:4326', scale=30)
                .unmask()
            )
            
            output_path = Path(self.config.RAW_DATA_DIR) / 'elevation.tif'
            
            # Export with error handling
            try:
                geemap.ee_export_image(
                    elevation,
                    filename=str(output_path),
                    scale=30,
                    region=self.roi,
                    crs='EPSG:4326'
                )
                self.logger.info(f"Successfully exported elevation data to {output_path}")
            except Exception as e:
                self.logger.error(f"Failed to export elevation data: {e}")
                return None
                
            return elevation
            
        except Exception as e:
            self.logger.error(f"Error collecting elevation data: {e}")
            return None

    def get_solar_radiation(self) -> Optional[ee.Image]:
        """
        Collect and process solar radiation data with improved temporal coverage.
        
        Returns:
            ee.Image: Processed radiation image or None if failed
        """
        try:
            # Get multiple years of data for better average
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * 3)  # 3 years of data
            
            gldas = (ee.ImageCollection('NASA/GLDAS/V021/NOAH/G025/T3H')
                .filterDate(start_date, end_date)
                .select(['SWdown_f_tavg'])
                .map(lambda img: img.multiply(11.574)  # Convert W/m^2 to MJ/m^2/day
                     .copyProperties(img, ['system:time_start'])))
            
            # Calculate seasonal averages
            seasons = {
                'winter': ['12', '01', '02'],
                'spring': ['03', '04', '05'],
                'summer': ['06', '07', '08'],
                'autumn': ['09', '10', '11']
            }
            
            seasonal_means = {}
            for season, months in seasons.items():
                seasonal = gldas.filter(
                    ee.Filter.calendarRange(int(months[0]), int(months[-1]), 'month')
                ).mean()
                seasonal_means[season] = seasonal
            
            # Calculate weighted annual average (more weight to summer)
            weights = {'winter': 0.2, 'spring': 0.25, 'summer': 0.35, 'autumn': 0.2}
            weighted_sum = ee.Image.constant(0)
            
            for season, image in seasonal_means.items():
                weighted_sum = weighted_sum.add(image.multiply(weights[season]))
            
            radiation = weighted_sum.clip(self.roi)
            
            output_path = Path(self.config.RAW_DATA_DIR) / 'solar_radiation.tif'
            
            try:
                geemap.ee_export_image(
                    radiation,
                    filename=str(output_path),
                    scale=250,
                    region=self.roi,
                    crs='EPSG:4326'
                )
                self.logger.info(f"Successfully exported solar radiation data to {output_path}")
            except Exception as e:
                self.logger.error(f"Failed to export solar radiation data: {e}")
                return None
                
            return radiation
            
        except Exception as e:
            self.logger.error(f"Error collecting solar radiation data: {e}")
            return None

    def get_land_cover(self) -> Optional[ee.Image]:
        """
        Collect and process land cover data with improved classification.
        
        Returns:
            ee.Image: Processed land cover image or None if failed
        """
        try:
            # Combine multiple land cover sources for better accuracy
            worldcover = ee.Image('ESA/WorldCover/v100')
            dynamic_world = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1').filterBounds(self.roi).first()
            
            # Remap classes to consistent scheme
            worldcover_remapped = worldcover.remap(
                [10, 20, 30, 40, 50, 60, 70, 80, 90, 95],  # Original classes
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]   # Remapped classes
            )
            
            dynamic_world_remapped = dynamic_world.select('label').remap(
                [0, 1, 2, 3, 4, 5, 6, 7, 8],  # Original classes
                [1, 2, 3, 4, 5, 6, 7, 8, 9]   # Remapped classes
            )
            
            # Combine classifications with weighted average
            combined = ee.Image.cat([worldcover_remapped, dynamic_world_remapped])
            landcover = combined.reduce(ee.Reducer.mode()).clip(self.roi)
            
            output_path = Path(self.config.RAW_DATA_DIR) / 'landcover.tif'
            
            try:
                geemap.ee_export_image(
                    landcover,
                    filename=str(output_path),
                    scale=10,
                    region=self.roi,
                    crs='EPSG:4326'
                )
                self.logger.info(f"Successfully exported land cover data to {output_path}")
            except Exception as e:
                self.logger.error(f"Failed to export land cover data: {e}")
                return None
                
            return landcover
            
        except Exception as e:
            self.logger.error(f"Error collecting land cover data: {e}")
            return None

    def collect_all_data(self) -> Optional[GEEData]:
        """
        Collect all required datasets in one operation.
        
        Returns:
            GEEData: Object containing all collected data or None if failed
        """
        try:
            elevation = self.get_elevation_data()
            solar_radiation = self.get_solar_radiation()
            land_cover = self.get_land_cover()
            
            if all([elevation, solar_radiation, land_cover]):
                return GEEData(
                    elevation=elevation,
                    solar_radiation=solar_radiation,
                    land_cover=land_cover,
                    file_paths={
                        'elevation': Path(self.config.RAW_DATA_DIR) / 'elevation.tif',
                        'solar_radiation': Path(self.config.RAW_DATA_DIR) / 'solar_radiation.tif',
                        'land_cover': Path(self.config.RAW_DATA_DIR) / 'landcover.tif'
                    }
                )
            else:
                self.logger.error("Failed to collect all required datasets")
                return None
                
        except Exception as e:
            self.logger.error(f"Error in collect_all_data: {e}")
            return None
        