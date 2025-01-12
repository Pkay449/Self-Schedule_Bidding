# src/main.py
"""
A top-level script to run the entire pipeline:
1) BADP main pipeline: trains BADP, generates offline data, evaluates.
2) NFQCA main pipeline: uses the offline data, trains NFQCA, evaluates.
"""

import logging
import warnings


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Import the pipeline entry points
from src.BADP.badp_main import badp_main
from src.Sequential_NFQCA.nfqca_main import nfqca_main

def main():
    """
    Orchestrates the entire project:
      - Step 1: Run BADP pipeline.
      - Step 2: Run NFQCA pipeline.
    """
    
    warnings.filterwarnings("ignore")

    logging.info("\n=== Starting BADP pipeline ===")
    badp_main()
    logging.info("\n=== BADP pipeline complete ===")

    logging.info("\n=== Starting NFQCA pipeline ===")
    nfqca_main()
    logging.info("\n=== NFQCA pipeline complete ===")

if __name__ == "__main__":
    main()
