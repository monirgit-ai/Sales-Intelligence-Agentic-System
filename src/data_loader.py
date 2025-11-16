"""
Data loading module for reading and preprocessing sales data.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
import logging

from .config import DATA_DIR, SAMPLE_DATA_FILE, DEFAULT_DATE_COLUMN

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles loading and preprocessing of sales data.
    
    Attributes:
        df: The loaded DataFrame
        file_path: Path to the data file
    """
    
    def __init__(self, file_path: Optional[Path] = None):
        """
        Initialize the DataLoader.
        
        Args:
            file_path: Path to the CSV file. If None, uses default sample data.
        """
        self.file_path = file_path or SAMPLE_DATA_FILE
        self.df: Optional[pd.DataFrame] = None
        
    def load(self) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Returns:
            DataFrame containing the sales data
            
        Raises:
            FileNotFoundError: If the data file doesn't exist
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.file_path}")
        
        logger.info(f"Loading data from {self.file_path}")
        self.df = pd.read_csv(self.file_path)
        
        # Convert date column to datetime if it exists
        if DEFAULT_DATE_COLUMN in self.df.columns:
            self.df[DEFAULT_DATE_COLUMN] = pd.to_datetime(
                self.df[DEFAULT_DATE_COLUMN],
                errors='coerce'
            )
        
        logger.info(f"Loaded {len(self.df)} rows and {len(self.df.columns)} columns")
        return self.df
    
    def get_data(self) -> pd.DataFrame:
        """
        Get the loaded DataFrame.
        
        Returns:
            The loaded DataFrame
            
        Raises:
            ValueError: If data hasn't been loaded yet
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load() first.")
        return self.df.copy()
    
    def validate_data(self) -> Dict[str, Any]:
        """
        Validate the loaded data structure.
        
        Returns:
            Dictionary containing validation results
        """
        if self.df is None:
            return {"valid": False, "error": "No data loaded"}
        
        required_columns = ["quantity", "revenue"]
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        validation_result = {
            "valid": len(missing_columns) == 0,
            "missing_columns": missing_columns,
            "row_count": len(self.df),
            "columns": list(self.df.columns),
            "date_range": None
        }
        
        if DEFAULT_DATE_COLUMN in self.df.columns:
            validation_result["date_range"] = {
                "min": self.df[DEFAULT_DATE_COLUMN].min(),
                "max": self.df[DEFAULT_DATE_COLUMN].max()
            }
        
        return validation_result

