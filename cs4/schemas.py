"""
CSV schema definitions and validation for CS4.
"""

from typing import List, Tuple, Optional
import pandas as pd
from abc import ABC, abstractmethod


class BaseSchema(ABC):
    """Base class for CSV schemas."""
    
    required_columns: List[str] = []
    optional_columns: List[str] = []
    
    @abstractmethod
    def validate_row(self, row: pd.Series) -> Tuple[bool, Optional[str]]:
        """Validate a single row. Returns (is_valid, error_message)."""
        pass
    
    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate entire DataFrame."""
        errors = []
        
        # Check required columns
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            return False, errors
        
        # Validate each row
        for idx, row in df.iterrows():
            is_valid, error = self.validate_row(row)
            if not is_valid:
                errors.append(f"Row {idx}: {error}")
        
        return len(errors) == 0, errors
    
    @property
    def all_columns(self) -> List[str]:
        """Get all columns (required + optional)."""
        return self.required_columns + self.optional_columns


class RawBlogsSchema(BaseSchema):
    """Schema for raw_blogs.csv"""
    
    required_columns = ["id", "url", "source", "text", "text_length"]
    optional_columns = ["created", "added"]
    
    def validate_row(self, row: pd.Series) -> Tuple[bool, Optional[str]]:
        # Check text_length is positive integer
        if not isinstance(row["text_length"], int) or row["text_length"] <= 0:
            return False, "text_length must be a positive integer"
        
        # Check text is not empty
        if not row["text"] or len(str(row["text"]).strip()) == 0:
            return False, "text cannot be empty"
        
        # Check URL format
        if not str(row["url"]).startswith(("http://", "https://")):
            return False, "url must start with http:// or https://"
        
        return True, None


class MergedBlogsSchema(BaseSchema):
    """Schema for merged_blogs.csv"""
    
    required_columns = [
        "original_blog_1", "original_blog_2", "merged_blog", 
        "similarity", "merge_type"
    ]
    optional_columns = ["blog1_id", "blog2_id", "model_used"]
    
    def validate_row(self, row: pd.Series) -> Tuple[bool, Optional[str]]:
        # Check similarity is float between 0 and 1
        if not isinstance(row["similarity"], (float, int)):
            return False, "similarity must be a number"
        if not 0 <= row["similarity"] <= 1:
            return False, "similarity must be between 0 and 1"
        
        # Check merge_type
        if row["merge_type"] not in ["similar", "dissimilar"]:
            return False, "merge_type must be 'similar' or 'dissimilar'"
        
        # Check blogs are not empty
        for col in ["original_blog_1", "original_blog_2", "merged_blog"]:
            if not row[col] or len(str(row[col]).strip()) == 0:
                return False, f"{col} cannot be empty"
        
        return True, None


class ConstraintsSchema(BaseSchema):
    """Schema for constraints.csv"""
    
    required_columns = [
        "instruction_number", "instruction", "main_task",
        "constraints", "model_used", "timestamp"
    ]
    optional_columns = ["tokens_used"]
    
    def validate_row(self, row: pd.Series) -> Tuple[bool, Optional[str]]:
        # Check instruction_number is positive integer
        if not isinstance(row["instruction_number"], int) or row["instruction_number"] <= 0:
            return False, "instruction_number must be a positive integer"
        
        # Check constraints format (should have 39 constraints)
        constraints_text = str(row["constraints"])
        # Count numbered constraints (format: "1. ", "2. ", etc.)
        import re
        constraint_count = len(re.findall(r'^\d+\.', constraints_text, re.MULTILINE))
        if constraint_count != 39:
            return False, f"Expected 39 constraints, found {constraint_count}"
        
        # Check main_task is not empty
        if not row["main_task"] or len(str(row["main_task"]).strip()) == 0:
            return False, "main_task cannot be empty"
        
        return True, None


class BaseGeneratedSchema(BaseSchema):
    """Schema for base_generated.csv"""
    
    required_columns = [
        "instruction_number", "main_task", "base_content",
        "content_length", "model_used", "tokens_used", "timestamp"
    ]
    optional_columns = ["temperature"]
    
    def validate_row(self, row: pd.Series) -> Tuple[bool, Optional[str]]:
        # Check instruction_number
        if not isinstance(row["instruction_number"], int) or row["instruction_number"] <= 0:
            return False, "instruction_number must be a positive integer"
        
        # Check content_length matches base_content
        if not isinstance(row["content_length"], int) or row["content_length"] <= 0:
            return False, "content_length must be a positive integer"
        
        actual_length = len(str(row["base_content"]))
        if abs(actual_length - row["content_length"]) > 10:  # Allow small discrepancy
            return False, f"content_length mismatch: {row['content_length']} vs {actual_length}"
        
        # Check tokens_used
        if not isinstance(row["tokens_used"], int) or row["tokens_used"] <= 0:
            return False, "tokens_used must be a positive integer"
        
        return True, None


class FittedContentSchema(BaseSchema):
    """Schema for fitted_content.csv"""
    
    required_columns = [
        "instruction_number", "main_task", "constraints", "base_content",
        "fitted_content", "fitted_length", "num_constraints",
        "model_used", "tokens_used", "timestamp"
    ]
    optional_columns = ["iterations"]
    
    def validate_row(self, row: pd.Series) -> Tuple[bool, Optional[str]]:
        # Check instruction_number
        if not isinstance(row["instruction_number"], int) or row["instruction_number"] <= 0:
            return False, "instruction_number must be a positive integer"
        
        # Check num_constraints is 39
        if row["num_constraints"] != 39:
            return False, f"num_constraints must be 39, got {row['num_constraints']}"
        
        # Check fitted_length
        if not isinstance(row["fitted_length"], int) or row["fitted_length"] <= 0:
            return False, "fitted_length must be a positive integer"
        
        actual_length = len(str(row["fitted_content"]))
        if abs(actual_length - row["fitted_length"]) > 10:
            return False, f"fitted_length mismatch: {row['fitted_length']} vs {actual_length}"
        
        return True, None


class EvaluationResultsSchema(BaseSchema):
    """Schema for evaluation_results.csv"""
    
    required_columns = [
        "instruction_number", "fitted_content", "constraints",
        "satisfaction_results", "num_satisfied", "total_constraints",
        "satisfaction_rate", "model_used", "tokens_used", "timestamp"
    ]
    optional_columns = []
    
    def validate_row(self, row: pd.Series) -> Tuple[bool, Optional[str]]:
        # Check instruction_number
        if not isinstance(row["instruction_number"], int) or row["instruction_number"] <= 0:
            return False, "instruction_number must be a positive integer"
        
        # Check total_constraints is 39
        if row["total_constraints"] != 39:
            return False, f"total_constraints must be 39, got {row['total_constraints']}"
        
        # Check num_satisfied is between 0 and 39
        if not 0 <= row["num_satisfied"] <= 39:
            return False, f"num_satisfied must be between 0 and 39, got {row['num_satisfied']}"
        
        # Check satisfaction_rate
        expected_rate = row["num_satisfied"] / row["total_constraints"]
        if abs(row["satisfaction_rate"] - expected_rate) > 0.01:
            return False, f"satisfaction_rate mismatch: {row['satisfaction_rate']} vs {expected_rate}"
        
        # Check satisfaction_results format
        results_text = str(row["satisfaction_results"])
        import re
        result_count = len(re.findall(r'^\d+\.\s+(Yes|No)\s+-', results_text, re.MULTILINE))
        if result_count != 39:
            return False, f"Expected 39 satisfaction results, found {result_count}"
        
        return True, None


# Validation functions
def validate_csv(csv_path: str, schema: BaseSchema) -> Tuple[bool, List[str]]:
    """Validate a CSV file against a schema."""
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
        return schema.validate(df)
    except Exception as e:
        return False, [f"Failed to read CSV: {str(e)}"]


def validate_raw_blogs_csv(csv_path: str) -> Tuple[bool, List[str]]:
    """Validate raw_blogs.csv"""
    return validate_csv(csv_path, RawBlogsSchema())


def validate_merged_blogs_csv(csv_path: str) -> Tuple[bool, List[str]]:
    """Validate merged_blogs.csv"""
    return validate_csv(csv_path, MergedBlogsSchema())


def validate_constraints_csv(csv_path: str) -> Tuple[bool, List[str]]:
    """Validate constraints.csv"""
    return validate_csv(csv_path, ConstraintsSchema())


def validate_base_generated_csv(csv_path: str) -> Tuple[bool, List[str]]:
    """Validate base_generated.csv"""
    return validate_csv(csv_path, BaseGeneratedSchema())


def validate_fitted_content_csv(csv_path: str) -> Tuple[bool, List[str]]:
    """Validate fitted_content.csv"""
    return validate_csv(csv_path, FittedContentSchema())


def validate_evaluation_results_csv(csv_path: str) -> Tuple[bool, List[str]]:
    """Validate evaluation_results.csv"""
    return validate_csv(csv_path, EvaluationResultsSchema())
