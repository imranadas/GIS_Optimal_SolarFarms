# src/processing/criteria.py

import logging
import rasterio
import numpy as np
import geopandas as gpd
from pathlib import Path
from scipy import ndimage
from rasterio import features
import concurrent.futures
from dataclasses import dataclass
from typing import Optional, Union

@dataclass
class ProcessedCriteria:
    """Data class to store processed criteria results"""
    slope: np.ndarray
    distances: dict
    normalized_layers: dict
    file_paths: dict

class CriteriaProcessor:
    """Processes and analyzes criteria for solar farm suitability analysis"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(config.PROCESSED_DATA_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def calculate_slope(self, elevation_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        Calculate slope from elevation data using improved algorithm.
        
        Args:
            elevation_path: Path to elevation data file
            
        Returns:
            np.ndarray: Calculated slope values or None if failed
        """
        try:
            with rasterio.open(elevation_path) as src:
                elevation = src.read(1)
                transform = src.transform
                
                # Calculate x and y spacing
                pixel_size_x = abs(transform[0])
                pixel_size_y = abs(transform[4])
                
                # Calculate gradients with better accuracy
                dy, dx = np.gradient(elevation, pixel_size_y, pixel_size_x)
                
                # Calculate slope using improved formula
                slope = np.degrees(np.arctan(np.sqrt(dx*dx + dy*dy)))
                
                # Apply terrain corrections
                slope = self._apply_terrain_corrections(slope, elevation)
                
                # Save slope raster
                output_path = self.output_dir / 'slope.tif'
                profile = src.profile.copy()
                
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(slope, 1)
                
                self.logger.info(f"Successfully calculated slope, saved to {output_path}")
                return slope
                
        except Exception as e:
            self.logger.error(f"Error calculating slope: {e}")
            return None

    def _apply_terrain_corrections(self, slope: np.ndarray, elevation: np.ndarray) -> np.ndarray:
        """Apply terrain-specific corrections to slope calculations"""
        # Correct for edge effects
        slope = ndimage.gaussian_filter(slope, sigma=1)
        
        # Remove spurious peaks
        slope = np.where(slope > 45, 45, slope)
        
        # Adjust for elevation-dependent factors
        elevation_factor = np.clip(elevation / 1000, 0.8, 1.2)
        slope = slope * elevation_factor
        
        return slope

    def calculate_distances(
        self, 
        target_path: Union[str, Path], 
        reference_features_path: Union[str, Path],
        max_distance: float = 10000  # Maximum distance in meters
    ) -> Optional[np.ndarray]:
        """
        Calculate distances to nearest features with improved efficiency.
        
        Args:
            target_path: Path to target raster
            reference_features_path: Path to reference features
            max_distance: Maximum distance to calculate
            
        Returns:
            np.ndarray: Distance raster or None if failed
        """
        try:
            # Read target raster
            with rasterio.open(target_path) as src:
                target = src.read(1)
                transform = src.transform
                profile = src.profile.copy()
                
                # Calculate pixel size in meters
                pixel_size = abs(transform[0])
                
            # Read reference features
            reference = gpd.read_file(reference_features_path)
            
            # Reproject if needed
            if reference.crs != profile['crs']:
                reference = reference.to_crs(profile['crs'])
            
            # Rasterize reference features
            mask = features.rasterize(
                [(geometry, 1) for geometry in reference.geometry],
                out_shape=target.shape,
                transform=transform,
                fill=0,
                dtype='uint8'
            )
            
            # Calculate distances using chunked processing for memory efficiency
            chunk_size = 1000
            distances = np.zeros_like(target, dtype=np.float32)
            
            def process_chunk(chunk_slice):
                return ndimage.distance_transform_edt(
                    ~mask[chunk_slice],
                    sampling=[pixel_size, pixel_size]
                )
            
            # Process in chunks using parallel processing
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for i in range(0, target.shape[0], chunk_size):
                    chunk_slice = slice(i, min(i + chunk_size, target.shape[0]))
                    futures.append(
                        executor.submit(process_chunk, chunk_slice)
                    )
                
                # Collect results
                for i, future in enumerate(futures):
                    start_idx = i * chunk_size
                    chunk_slice = slice(
                        start_idx,
                        min(start_idx + chunk_size, target.shape[0])
                    )
                    distances[chunk_slice] = future.result()
            
            # Cap maximum distance
            distances = np.minimum(distances, max_distance)
            
            # Save distance raster
            output_path = self.output_dir / f"distances_{Path(reference_features_path).stem}.tif"
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(distances, 1)
            
            self.logger.info(f"Successfully calculated distances, saved to {output_path}")
            return distances
            
        except Exception as e:
            self.logger.error(f"Error calculating distances: {e}")
            return None

    def normalize_raster(
        self, 
        raster_data: np.ndarray,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        method: str = 'minmax'
    ) -> np.ndarray:
        """
        Normalize raster data using various methods.
        
        Args:
            raster_data: Input raster data
            min_val: Optional minimum value for normalization
            max_val: Optional maximum value for normalization
            method: Normalization method ('minmax', 'zscore', 'robust')
            
        Returns:
            np.ndarray: Normalized raster data
        """
        try:
            if method == 'minmax':
                if min_val is None:
                    min_val = np.nanpercentile(raster_data, 1)
                if max_val is None:
                    max_val = np.nanpercentile(raster_data, 99)
                    
                normalized = (raster_data - min_val) / (max_val - min_val)
                normalized = np.clip(normalized, 0, 1)
                
            elif method == 'zscore':
                mean = np.nanmean(raster_data)
                std = np.nanstd(raster_data)
                normalized = (raster_data - mean) / std
                
                # Scale to 0-1 range
                normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min())
                
            elif method == 'robust':
                q1 = np.nanpercentile(raster_data, 25)
                q3 = np.nanpercentile(raster_data, 75)
                iqr = q3 - q1
                
                normalized = (raster_data - q1) / iqr
                normalized = np.clip(normalized, 0, 1)
                
            else:
                raise ValueError(f"Unsupported normalization method: {method}")
            
            return normalized
            
        except Exception as e:
            self.logger.error(f"Error normalizing raster: {e}")
            return None

    def process_all_criteria(self) -> Optional[ProcessedCriteria]:
        """
        Process all criteria in one operation
        
        Returns:
            ProcessedCriteria: Object containing all processed criteria or None if failed
        """
        try:
            # Calculate slope
            slope = self.calculate_slope(
                self.config.RAW_DATA_DIR / 'elevation.tif'
            )
            
            # Calculate distances for different features
            distances = {}
            for feature in ['roads', 'power_infrastructure', 'residential_areas']:
                distances[feature] = self.calculate_distances(
                    self.config.RAW_DATA_DIR / 'elevation.tif',
                    self.config.RAW_DATA_DIR / f'{feature}.gpkg'
                )
            
            # Normalize all layers
            normalized_layers = {}
            for name, layer in {
                'slope': slope,
                **distances
            }.items():
                normalized_layers[name] = self.normalize_raster(
                    layer,
                    method='robust' if name == 'slope' else 'minmax'
                )
            
            return ProcessedCriteria(
                slope=slope,
                distances=distances,
                normalized_layers=normalized_layers,
                file_paths={
                    'slope': self.output_dir / 'slope.tif',
                    **{
                        f'distances_{k}': self.output_dir / f'distances_{k}.tif'
                        for k in distances.keys()
                    }
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error processing criteria: {e}")
            return None
