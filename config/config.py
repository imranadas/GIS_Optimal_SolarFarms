# config/config.py

import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class ProjectConfig:
    """Configuration for solar farm site selection analysis"""
    
    def __init__(self, config_path: str = None):
        # Load custom config if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                custom_config = yaml.safe_load(f)
        else:
            custom_config = {}
            
        # Region of Interest bounds
        self.ROI_BOUNDS = custom_config.get('roi_bounds', {
            'north': 27.25,
            'south': 25.25,
            'east': 44.50,
            'west': 42.50
        })
        
        # Criteria weights from research paper
        self.CRITERIA_WEIGHTS = custom_config.get('criteria_weights', {
            'ghi': 0.55,
            'residential_proximity': 0.21,
            'road_proximity': 0.10,
            'powerline_proximity': 0.11,
            'slope': 0.03
        })
        
        # Suitability classification thresholds
        self.SUITABILITY_THRESHOLDS = custom_config.get('suitability_thresholds', {
            'most_suitable': 0.8,
            'suitable': 0.6,
            'moderately_suitable': 0.4,
            'unsuitable': 0.2
        })
        
        # Analysis parameters
        self.ANALYSIS_PARAMS = custom_config.get('analysis_params', {
            'min_area': 10000,  # minimum area in m² for suitable sites
            'resolution': 30,   # spatial resolution in meters
            'buffer_distances': {  # buffer distances in meters
                'roads': 100,
                'powerlines': 500,
                'residential': 1000
            },
            'slope_threshold': 15  # maximum slope in degrees
        })
        
        # Paths
        self.BASE_DIR = Path(custom_config.get('base_dir', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.RAW_DATA_DIR = self.BASE_DIR / 'data' / 'raw'
        self.PROCESSED_DATA_DIR = self.BASE_DIR / 'data' / 'processed'
        self.OUTPUT_DIR = self.BASE_DIR / 'output'
        self.LOGS_DIR = self.BASE_DIR / 'logs'
        
        # Create directories
        for directory in [self.RAW_DATA_DIR, self.PROCESSED_DATA_DIR, 
                         self.OUTPUT_DIR, self.LOGS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # API credentials (should be moved to environment variables in production)
        self.CREDENTIALS = {
            'gee_service_account': os.getenv('GEE_SERVICE_ACCOUNT', ''),
            'nasa_username': os.getenv('NASA_USERNAME', ''),
            'nasa_password': os.getenv('NASA_PASSWORD', '')
        }
        
    def save_config(self, filepath: str):
        """Save current configuration to YAML file"""
        config_dict = {
            'roi_bounds': self.ROI_BOUNDS,
            'criteria_weights': self.CRITERIA_WEIGHTS,
            'suitability_thresholds': self.SUITABILITY_THRESHOLDS,
            'analysis_params': self.ANALYSIS_PARAMS
        }
        
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f)
            
    @classmethod
    def load_config(cls, filepath: str) -> 'ProjectConfig':
        """Load configuration from YAML file"""
        return cls(config_path=filepath)
