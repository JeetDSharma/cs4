"""
Constraint satisfaction evaluator module.
Converted from constraint_satisfaction.py
"""

import pandas as pd
import logging
import re
from time import sleep
from typing import Optional, Tuple
from datetime import datetime

from cs4.core.prompts import get_evaluation_prompt
from cs4.utils.llm_client import OpenAIClient, AnthropicClient
from cs4.config import Config


class ConstraintEvaluator:
    """Evaluate constraint satisfaction in generated content."""
    
    def __init__(
        self,
        llm_client: Optional[object] = None,
        model: str = None,
        content_type: str = "blog",
        retry_attempts: int = 3,
        delay: float = 1.0
    ):
        """
        Initialize constraint evaluator.
        
        Args:
            llm_client: LLM client (OpenAI or Anthropic)
            model: Model identifier
            content_type: Type of content (blog, story, news)
            retry_attempts: Number of retry attempts on failure
            delay: Delay in seconds between retries
        """
        self.llm_client = llm_client or OpenAIClient(log_usage=True)
        self.model = model or Config.DEFAULT_EVALUATION_MODEL
        self.content_type = content_type
        self.retry_attempts = retry_attempts
        self.delay = delay
        
        self.logger = logging.getLogger("CS4Evaluator")
    
    def evaluate_content(
        self,
        content: str,
        constraints: str,
        log: bool = True
    ) -> Tuple[str, int, int]:
        """
        Evaluate content against constraints.
        
        Args:
            content: Generated content to evaluate
            constraints: Newline-separated list of constraints
            log: Whether to log token usage
            
        Returns:
            Tuple of (satisfaction_results, num_satisfied, tokens_used)
        """
        prompt = get_evaluation_prompt(
            content_type=self.content_type,
            content=content,
            constraints=constraints
        )
        
        for attempt in range(1, self.retry_attempts + 1):
            try:
                if isinstance(self.llm_client, OpenAIClient):
                    response = self.llm_client.chat_completion(
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        model=self.model
                    )
                    results = response.choices[0].message.content.strip()
                    tokens = response.usage.total_tokens
                elif isinstance(self.llm_client, AnthropicClient):
                    response = self.llm_client.create_message(
                        messages=[{"role": "user", "content": prompt}],
                        model=self.model    
                    )
                    results = response.content[0].text
                    tokens = response.usage.input_tokens + response.usage.output_tokens
                else:
                    raise ValueError("Unknown client type")
                
                # Extract number of satisfied constraints
                num_satisfied = self._extract_satisfaction_count(results)
                
                if log:
                    self.logger.info(f"Total tokens used: {tokens}")
                    self.logger.info(f"Constraints satisfied: {num_satisfied}")
                
                return results, num_satisfied, tokens
                
            except Exception as e:
                self.logger.warning(
                    f"Attempt {attempt}/{self.retry_attempts} failed: {e}"
                )
                if attempt < self.retry_attempts:
                    sleep(self.delay)
                else:
                    raise
        
        raise RuntimeError("Failed to evaluate content")
    
    def _extract_satisfaction_count(self, results: str) -> int:
        """Extract the number of satisfied constraints from evaluation results."""
        match = re.search(r'Number of constraints satisfied:\s*(\d+)', results)
        if match:
            return int(match.group(1))
        
        # Fallback: count "Yes" occurrences
        yes_count = len(re.findall(r'^\d+\.\s+Yes', results, re.MULTILINE))
        return yes_count
    
    def evaluate_batch(
        self,
        df: pd.DataFrame,
        content_column: str = "fitted_content",
        constraints_column: str = "constraints",
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Evaluate constraint satisfaction for a batch of samples.
        
        Args:
            df: Input DataFrame (typically fitted_content.csv)
            content_column: Name of column with content to evaluate
            constraints_column: Name of column with constraints
            output_path: Optional path to save results
            
        Returns:
            DataFrame with evaluation results
        """
        if content_column not in df.columns:
            raise ValueError(f"Column '{content_column}' not found in DataFrame")
        if constraints_column not in df.columns:
            raise ValueError(f"Column '{constraints_column}' not found in DataFrame")
        
        # Check for instruction_number
        if "instruction_number" in df.columns:
            has_instruction_num = True
        else:
            has_instruction_num = False
            self.logger.warning("No 'instruction_number' column found, using index")
        
        self.logger.info(f"Evaluating {len(df)} samples")
        
        results = []
        for idx, row in df.iterrows():
            content = row[content_column]
            constraints = row[constraints_column]
            instruction_num = row["instruction_number"] if has_instruction_num else idx + 1
            
            self.logger.info(f"Evaluating sample #{instruction_num}")
            
            try:
                satisfaction_results, num_satisfied, tokens = self.evaluate_content(
                    content=content,
                    constraints=constraints,
                    log=True
                )
                
                # Count total constraints
                total_constraints = len(re.findall(r'^\d+\.', constraints, re.MULTILINE))
                satisfaction_rate = num_satisfied / total_constraints if total_constraints > 0 else 0.0
                
                results.append({
                    "instruction_number": instruction_num,
                    "fitted_content": content,
                    "constraints": constraints,
                    "satisfaction_results": satisfaction_results,
                    "num_satisfied": num_satisfied,
                    "total_constraints": total_constraints,
                    "satisfaction_rate": satisfaction_rate,
                    "model_used": self.model,
                    "tokens_used": tokens,
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                self.logger.error(
                    f"Failed to evaluate sample {instruction_num}: {e}"
                )
                results.append({
                    "instruction_number": instruction_num,
                    "fitted_content": content,
                    "constraints": constraints,
                    "satisfaction_results": "",
                    "num_satisfied": 0,
                    "total_constraints": 0,
                    "satisfaction_rate": 0.0,
                    "model_used": self.model,
                    "tokens_used": 0,
                    "timestamp": datetime.now().isoformat()
                })
        
        result_df = pd.DataFrame(results)
        
        if output_path:
            result_df.to_csv(output_path, index=False, encoding="utf-8")
            self.logger.info(f"Evaluation results saved to {output_path}")
        
        return result_df


# Legacy function for backward compatibility
def evaluate_constraints(
    input_path: str,
    output_path: str,
    model: str = None,
    content_type: str = "blog"
):
    """
    Legacy interface for constraint evaluation.
    
    Args:
        input_path: Path to CSV with generated content
        output_path: Path to save evaluation results
        model: LLM model identifier
        content_type: Type of content (blog, story, news)
    """
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    evaluator = ConstraintEvaluator(model=model, content_type=content_type)
    df = pd.read_csv(input_path, encoding="utf-8")
    
    result_df = evaluator.evaluate_batch(df, output_path=output_path)
    
    return result_df
