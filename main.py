# main.py

import json
import logging
import numpy as np
import logging.handlers
import concurrent.futures
from datetime import datetime
from typing import Optional, Dict, Any
from config.config import ProjectConfig
from src.visualization.map_visualizer import MapVisualizer
from src.data_collection.gee_collector import GEEDataCollector
from src.data_collection.osm_collector import OSMDataCollector
from src.processing.criteria import CriteriaProcessor, ProcessedCriteria
from src.processing.suitability import SuitabilityAnalyzer, SuitabilityResult


class SolarFarmAnalysis:
    """Main workflow manager for solar farm site selection analysis"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Initialize configuration
        self.config = ProjectConfig.load_config(config_path) if config_path else ProjectConfig()
        
        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.gee_collector = GEEDataCollector(self.config)
        self.osm_collector = OSMDataCollector(self.config)
        self.criteria_processor = CriteriaProcessor(self.config)
        self.suitability_analyzer = SuitabilityAnalyzer(self.config)
        self.map_visualizer = MapVisualizer(self.config)
        
    def setup_logging(self):
        """Setup logging configuration"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.config.LOGS_DIR / f'solar_farm_analysis_{timestamp}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),
                logging.handlers.RotatingFileHandler(
                    self.config.LOGS_DIR / 'solar_farm_analysis.log',
                    maxBytes=5*1024*1024,  # 5MB
                    backupCount=5
                )
            ]
        )

    def collect_data(self) -> Optional[Dict[str, Any]]:
        """Collect all required data"""
        self.logger.info("Starting data collection...")
        
        try:
            # Collect data in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Earth Engine data
                gee_future = executor.submit(self.gee_collector.collect_all_data)
                
                # OSM data
                osm_futures = {
                    'roads': executor.submit(self.osm_collector.get_road_network),
                    'power': executor.submit(self.osm_collector.get_power_infrastructure),
                    'residential': executor.submit(self.osm_collector.get_residential_areas)
                }
                
                # Wait for results
                gee_data = gee_future.result()
                osm_data = {k: v.result() for k, v in osm_futures.items()}
                
            if not gee_data or not all(osm_data.values()):
                raise ValueError("Failed to collect all required data")
                
            self.logger.info("Data collection completed successfully")
            return {'gee': gee_data, 'osm': osm_data}
            
        except Exception as e:
            self.logger.error(f"Error in data collection: {e}")
            return None

    def process_criteria(self, data: Dict[str, Any]) -> Optional[ProcessedCriteria]:
        """Process all criteria layers"""
        self.logger.info("Processing criteria layers...")
        
        try:
            processed = self.criteria_processor.process_all_criteria()
            
            if not processed:
                raise ValueError("Failed to process criteria")
                
            self.logger.info("Criteria processing completed successfully")
            return processed
            
        except Exception as e:
            self.logger.error(f"Error in criteria processing: {e}")
            return None

    def analyze_suitability(
        self,
        processed_criteria: ProcessedCriteria
    ) -> Optional[SuitabilityResult]:
        """Analyze site suitability"""
        self.logger.info("Analyzing site suitability...")
        
        try:
            # Prepare criteria layers
            criteria_layers = [
                processed_criteria.normalized_layers[name]
                for name in self.config.CRITERIA_WEIGHTS.keys()
            ]
            
            # Calculate restricted areas mask
            mask = self._calculate_restricted_areas(processed_criteria)
            
            # Perform analysis
            result = self.suitability_analyzer.analyze_site_suitability(
                criteria_layers,
                mask=mask
            )
            
            if not result:
                raise ValueError("Failed to analyze suitability")
                
            self.logger.info("Suitability analysis completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in suitability analysis: {e}")
            return None

    def _calculate_restricted_areas(
        self,
        processed_criteria: ProcessedCriteria
    ) -> np.ndarray:
        """Calculate mask for restricted areas"""
        try:
            # Combine all restriction criteria
            mask = np.ones_like(processed_criteria.slope, dtype=bool)
            
            # Apply slope threshold
            mask &= processed_criteria.slope <= self.config.ANALYSIS_PARAMS['slope_threshold']
            
            # Apply minimum distances
            for feature, distance in self.config.ANALYSIS_PARAMS['buffer_distances'].items():
                if feature in processed_criteria.distances:
                    mask &= processed_criteria.distances[feature] >= distance
                    
            return mask
            
        except Exception as e:
            self.logger.error(f"Error calculating restricted areas: {e}")
            return None

    def visualize_results(
        self,
        suitability_result: SuitabilityResult
    ) -> Optional[Dict[str, str]]:
        """Create and save visualizations"""
        self.logger.info("Creating visualizations...")
        
        try:
            # Create interactive map
            suitability_map = self.map_visualizer.create_suitability_map(
                suitability_result.continuous,
                suitability_result.classified
            )
            
            # Save map
            map_path = self.config.OUTPUT_DIR / 'suitability_map.html'
            self.map_visualizer.save_map(suitability_map, str(map_path))
            
            # Save statistics
            stats_path = self.config.OUTPUT_DIR / 'analysis_statistics.json'
            with open(stats_path, 'w') as f:
                json.dump(suitability_result.statistics, f, indent=4)
                
            self.logger.info("Visualization completed successfully")
            return {
                'map': str(map_path),
                'statistics': str(stats_path)
            }
            
        except Exception as e:
            self.logger.error(f"Error in visualization: {e}")
            return None

    def run_analysis(self) -> bool:
        """Run complete analysis workflow"""
        try:
            # Collect data
            data = self.collect_data()
            if not data:
                return False
                
            # Process criteria
            processed_criteria = self.process_criteria(data)
            if not processed_criteria:
                return False
                
            # Analyze suitability
            suitability_result = self.analyze_suitability(processed_criteria)
            if not suitability_result:
                return False
                
            # Create visualizations
            visualization_paths = self.visualize_results(suitability_result)
            if not visualization_paths:
                return False
                
            self.logger.info("Analysis completed successfully")
            self.logger.info(f"Results saved to: {visualization_paths}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in analysis workflow: {e}")
            return False

def main():
    """Main entry point"""
    try:
        # Initialize and run analysis
        analysis = SolarFarmAnalysis()
        success = analysis.run_analysis()
        
        if not success:
            logging.error("Analysis failed")
            return 1
        return 0
        
    except Exception as e:
        logging.error(f"Fatal error in main execution: {e}")
        return 1

if __name__ == '__main__':
    exit(main())
