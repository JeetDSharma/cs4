"""
Constraint expansion module - creates subsets of constraints for testing.
"""

import pandas as pd
import re
import logging
from typing import List, Optional


class ConstraintExpander:
    """Expand constraints into progressive subsets (buckets)."""
    
    def __init__(self, subset_sizes: List[int] = None):
        """
        Initialize constraint expander.
        
        Args:
            subset_sizes: List of constraint counts to include for each subset
                         Default: [7, 15, 23, 31, 39]
        """
        self.subset_sizes = subset_sizes or [7, 15, 23, 31, 39]
        self.logger = logging.getLogger("CS4Generator")
    
    def expand_constraints(
        self,
        df: pd.DataFrame,
        constraint_column: str = "constraints",
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Expand each row by creating copies with progressively larger constraint subsets.
        
        For each original row, creates N new rows (one per subset_size),
        where each row has a subset of the original constraints.
        
        Args:
            df: Input DataFrame with constraints column
            constraint_column: Name of column containing constraints
            output_path: Optional path to save expanded DataFrame
            
        Returns:
            Expanded DataFrame with 'selected_constraints' and 'subset_size' columns
        """
        if constraint_column not in df.columns:
            raise ValueError(f"Input DataFrame must contain '{constraint_column}' column")
        
        self.logger.info(f"Expanding {len(df)} rows into constraint buckets")
        self.logger.info(f"Subset sizes: {self.subset_sizes}")
        
        expanded_rows = []
        
        for idx, row in df.iterrows():
            constraints_text = row[constraint_column]
            
            # Parse constraints from text
            constraints_list = self._parse_constraints(constraints_text)
            total_constraints = len(constraints_list)
            
            instruction_num = row.get("instruction_number", idx + 1)
            self.logger.info(f"Instruction #{instruction_num}: Found {total_constraints} constraints")
            
            # Create a subset for each size
            for size in self.subset_sizes:
                subset = constraints_list[:min(size, total_constraints)]
                
                # Re-number constraints consistently
                selected_text = "\n".join(f"{i+1}. {subset[i]}" for i in range(len(subset)))
                
                # Create new row with subset
                new_row = row.copy()
                new_row["selected_constraints"] = selected_text
                new_row["subset_size"] = min(size, total_constraints)
                expanded_rows.append(new_row)
        
        expanded_df = pd.DataFrame(expanded_rows)
        
        self.logger.info(f"Expanded to {len(expanded_df)} rows ({len(df)} × {len(self.subset_sizes)} subsets)")
        
        if output_path:
            expanded_df.to_csv(output_path, index=False, encoding="utf-8")
            self.logger.info(f"Expanded constraints saved to {output_path}")
        
        return expanded_df
    
    def _parse_constraints(self, constraints_text: str) -> List[str]:
        """
        Parse constraints text into a list of individual constraints.
        
        Handles formats like:
        - "Constraints:\n1. First\n2. Second"
        - "1. First\n2. Second"
        
        Args:
            constraints_text: Text containing numbered constraints
            
        Returns:
            List of constraint strings (without numbers)
        """
        # Remove "Constraints:" prefix if present
        constraints_text = re.sub(
            r"^Constraints:\s*", 
            "", 
            constraints_text.strip(), 
            flags=re.IGNORECASE
        )
        
        # Split on numeric list markers (e.g., "1.", "2.", ...)
        constraints_list = re.split(r'\n\s*\d+\.\s*', constraints_text)
        
        # Clean up each constraint
        constraints_list = [
            re.sub(r'^\d+\.\s*', '', c).strip() 
            for c in constraints_list 
            if c.strip()
        ]
        
        return constraints_list
