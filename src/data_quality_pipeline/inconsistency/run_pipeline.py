import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_quality_pipeline.inconsistency.config import SYSTEMS_CONFIG, PipelineConfig
from data_quality_pipeline.inconsistency.pipeline import InconsistencyPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # Load environment variables
        load_dotenv()
        
        # Initialize pipeline configuration
        config = PipelineConfig(
            systems=SYSTEMS_CONFIG,
            max_workers=4
        )
        
        # Create and run pipeline
        pipeline = InconsistencyPipeline(config)
        findings = pipeline.run_pipeline()
        
        if not findings:
            logger.info("No findings generated")
            
    except Exception as e:
        logger.error(f"Error running pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 