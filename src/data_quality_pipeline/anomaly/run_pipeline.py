#!/usr/bin/env python3
"""
Anomaly Detection Pipeline Runner

This module provides the main entry point for running the anomaly detection pipeline.
It can be executed directly or imported as a module.

Usage:
    python -m src.data_quality_pipeline.anomaly.run_pipeline
    or
    python src/data_quality_pipeline/anomaly/run_pipeline.py
"""

import logging
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_quality_pipeline.anomaly.pipeline import AnomalyPipeline

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('anomaly_detection.log')
        ]
    )

def main():
    """Main function to run the anomaly detection pipeline."""
    try:
        # Setup logging
        setup_logging()
        logger = logging.getLogger(__name__)
        
        logger.info("=" * 60)
        logger.info("STARTING ANOMALY DETECTION PIPELINE")
        logger.info("=" * 60)
        
        # Create and run pipeline
        pipeline = AnomalyPipeline()
        findings = pipeline.run()
        
        logger.info("=" * 60)
        logger.info("ANOMALY DETECTION PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"Total findings generated: {len(findings)}")
        logger.info("=" * 60)
        
        return findings
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install pandas numpy scikit-learn")
        sys.exit(1)
    except Exception as e:
        print(f"Error running anomaly detection pipeline: {e}")
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 