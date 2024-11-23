# src/processing/suitability.py

import logging
import rasterio
import numpy as np
from pathlib import Path
from scipy import ndimage
import concurrent.futures
from dataclasses import dataclass
from typing import Optional, List, Dict

@dataclass
class SuitabilityResult:
    """Data class to store suitability analysis results"""
    continuous: np.ndarray
    classified: np.ndarray
    statistics: Dict[str, float]
    file_paths: Dict[str, Path]

class SuitabilityAnalyzer:
    """Analyzes and classifies site suitability for solar farms"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(config.PROCESSED_DATA_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def calculate_suitability(
        self,
        criteria_layers: List[np.ndarray],
        weights: Optional[List[float]] = None,
        mask: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        Calculate overall suitability using weighted overlay with advanced options.
        
        Args:
            criteria_layers: List of criteria layers
            weights: Optional list of weights (uses config weights if None)
            mask: Optional mask for restricted areas
            
        Returns:
            np.ndarray: Suitability scores or None if failed
        """
        try:
            if weights is None:
                weights = list(self.config.CRITERIA_WEIGHTS.values())
                
            if len(criteria_layers) != len(weights):
                raise ValueError("Number of layers must match number of weights")
                
            # Initialize suitability array
            suitability = np.zeros_like(criteria_layers[0], dtype=np.float32)
            
            # Calculate weighted sum with parallel processing
            chunk_size = 1000
            
            def process_chunk(chunk_slice):
                chunk_result = np.zeros_like(
                    criteria_layers[0][chunk_slice],
                    dtype=np.float32
                )
                for layer, weight in zip(criteria_layers, weights):
                    chunk_result += layer[chunk_slice] * weight
                return chunk_result
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for i in range(0, suitability.shape[0], chunk_size):
                    chunk_slice = slice(i, min(i + chunk_size, suitability.shape[0]))
                    futures.append(
                        executor.submit(process_chunk, chunk_slice)
                    )
                
                # Collect results
                for i, future in enumerate(futures):
                    start_idx = i * chunk_size
                    chunk_slice = slice(
                        start_idx,
                        min(start_idx + chunk_size, suitability.shape[0])
                    )
                    suitability[chunk_slice] = future.result()
            
            # Apply mask if provided
            if mask is not None:
                suitability = np.where(mask, suitability, np.nan)
            
            # Normalize result
            suitability = self._normalize_suitability(suitability)
            
            # Apply smoothing to reduce noise
            suitability = ndimage.gaussian_filter(suitability, sigma=1)
            
            return suitability
            
        except Exception as e:
            self.logger.error(f"Error calculating suitability: {e}")
            return None

    def _normalize_suitability(self, suitability: np.ndarray) -> np.ndarray:
        """Normalize suitability scores with robust scaling"""
        valid_mask = ~np.isnan(suitability)
        if not np.any(valid_mask):
            return suitability
            
        valid_data = suitability[valid_mask]
        
        # Use percentile-based normalization
        p1 = np.percentile(valid_data, 1)
        p99 = np.percentile(valid_data, 99)
        
        normalized = np.zeros_like(suitability)
        normalized[valid_mask] = np.clip(
            (valid_data - p1) / (p99 - p1),
            0, 1
        )
        normalized[~valid_mask] = np.nan
        
        return normalized

    def classify_suitability(
        self,
        suitability: np.ndarray,
        thresholds: Optional[Dict[str, float]] = None,
        min_area: float = 10000  # minimum area in m²
    ) -> Optional[np.ndarray]:
        """
        Classify suitability into categories with advanced processing.
        
        Args:
            suitability: Suitability scores
            thresholds: Optional classification thresholds
            min_area: Minimum area for valid regions
            
        Returns:
            np.ndarray: Classified suitability or None if failed
        """
        try:
            if thresholds is None:
                thresholds = self.config.SUITABILITY_THRESHOLDS
            
            classified = np.zeros_like(suitability, dtype=np.uint8)
            
            # Apply classification thresholds
            classified[suitability >= thresholds['most_suitable']] = 4
            classified[
                (suitability >= thresholds['suitable']) &
                (suitability < thresholds['most_suitable'])
            ] = 3
            classified[
                (suitability >= thresholds['moderately_suitable']) &
                (suitability < thresholds['suitable'])
            ] = 2
            classified[
                (suitability >= thresholds['unsuitable']) &
                (suitability < thresholds['moderately_suitable'])
            ] = 1
            
            # Remove small isolated areas
            for class_value in range(1, 5):
                class_mask = classified == class_value
                labeled, num_features = ndimage.label(class_mask)
                
                if num_features > 0:
                    # Calculate area of each feature
                    feature_sizes = ndimage.sum(
                        class_mask,
                        labeled,
                        range(1, num_features + 1)
                    )
                    
                    # Remove features smaller than minimum area
                    small_features = feature_sizes < min_area
                    remove_pixels = np.in1d(labeled.ravel(), 
                                          np.where(small_features)[0] + 1).reshape(labeled.shape)
                    classified[remove_pixels] = 0
            
            # Smooth boundaries between classes
            classified = self._smooth_boundaries(classified)
            
            return classified
            
        except Exception as e:
            self.logger.error(f"Error classifying suitability: {e}")
            return None

    def _smooth_boundaries(self, classified: np.ndarray) -> np.ndarray:
        """Smooth boundaries between suitability classes"""
        smoothed = classified.copy()
        
        # Apply majority filter to smooth boundaries
        for _ in range(2):  # Apply twice for better smoothing
            smoothed = ndimage.median_filter(smoothed, size=3)
        
        return smoothed

    def calculate_statistics(
        self,
        suitability: np.ndarray,
        classified: np.ndarray
    ) -> Dict[str, float]:
        """Calculate statistics for suitability analysis results"""
        try:
            stats = {}
            
            # Calculate area statistics
            pixel_area = 900  # Assuming 30m resolution (30*30 = 900 m²)
            total_area = np.sum(~np.isnan(suitability)) * pixel_area
            
            for class_value, class_name in [
                (4, 'most_suitable'),
                (3, 'suitable'),
                (2, 'moderately_suitable'),
                (1, 'unsuitable'),
                (0, 'restricted')
            ]:
                class_pixels = np.sum(classified == class_value)
                class_area = class_pixels * pixel_area
                stats[f'{class_name}_area'] = class_area
                stats[f'{class_name}_percentage'] = (class_area / total_area) * 100
            
            # Calculate additional statistics
            stats['mean_suitability'] = np.nanmean(suitability)
            stats['median_suitability'] = np.nanmedian(suitability)
            stats['std_suitability'] = np.nanstd(suitability)
            
            # Calculate potential solar capacity
            # Assuming 4 hectares per MW for utility-scale solar farms
            suitable_area_ha = (stats['most_suitable_area'] + stats['suitable_area']) / 10000
            stats['potential_capacity_mw'] = suitable_area_ha / 4
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating statistics: {e}")
            return {}

    def analyze_site_suitability(
        self,
        criteria_layers: List[np.ndarray],
        mask: Optional[np.ndarray] = None
    ) -> Optional[SuitabilityResult]:
        """
        Perform complete suitability analysis
        
        Args:
            criteria_layers: List of criteria layers
            mask: Optional mask for restricted areas
            
        Returns:
            SuitabilityResult: Complete analysis results or None if failed
        """
        try:
            # Calculate suitability
            suitability = self.calculate_suitability(
                criteria_layers,
                mask=mask
            )
            
            if suitability is None:
                return None
            
            # Classify suitability
            classified = self.classify_suitability(suitability)
            
            if classified is None:
                return None
            
            # Calculate statistics
            statistics = self.calculate_statistics(suitability, classified)
            
            # Save results
            output_paths = {}
            
            # Save continuous suitability
            suitability_path = self.output_dir / 'suitability_continuous.tif'
            with rasterio.open(criteria_layers[0], 'r') as src:
                profile = src.profile.copy()
                profile.update(dtype='float32', count=1)
                with rasterio.open(suitability_path, 'w', **profile) as dst:
                    dst.write(suitability, 1)
            output_paths['continuous'] = suitability_path
            
            # Save classified suitability
            classified_path = self.output_dir / 'suitability_classified.tif'
            profile.update(dtype='uint8')
            with rasterio.open(classified_path, 'w', **profile) as dst:
                dst.write(classified, 1)
            output_paths['classified'] = classified_path
            
            return SuitabilityResult(
                continuous=suitability,
                classified=classified,
                statistics=statistics,
                file_paths=output_paths
            )
            
        except Exception as e:
            self.logger.error(f"Error in site suitability analysis: {e}")
            return None
