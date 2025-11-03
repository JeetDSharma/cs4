"""
Constraint fitting module - fits base content to satisfy constraints.
Converted from fit_base_blog.ipynb
"""

import pandas as pd
import logging
from time import sleep
from typing import Optional
from datetime import datetime

from cs4.core.prompts import get_constraint_fitting_prompt
from cs4.utils.llm_client import OpenAIClient, AnthropicClient
from cs4.config import Config


class ConstraintFitter:
    """Fit base content to satisfy multiple constraints."""
    
    def __init__(
        self,
        llm_client: Optional[object] = None,
        model: str = "gpt-4-mini",
        content_type: str = "blog",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        retry_attempts: int = 3,
        delay: float = 1.0
    ):
        """
        Initialize constraint fitter.
        
        Args:
            llm_client: LLM client (OpenAI or Anthropic)
            model: Model identifier
            content_type: Type of content (blog, story, news)
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            retry_attempts: Number of retry attempts on failure
            delay: Delay in seconds between retries
        """
        self.llm_client = llm_client or OpenAIClient(log_usage=True)
        self.model = model
        self.content_type = content_type
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retry_attempts = retry_attempts
        self.delay = delay
        
        self.logger = logging.getLogger("CS4Generator")
    
    def fit_content(
        self,
        task: str,
        base_content: str,
        constraints: str,
        log: bool = True
    ) -> tuple[str, int]:
        """
        Fit base content to satisfy all constraints.
        
        Args:
            task: Task description
            base_content: Base content to fit
            constraints: Newline-separated list of constraints
            log: Whether to log token usage
            
        Returns:
            Tuple of (fitted_content, tokens_used)
        """
        prompt = get_constraint_fitting_prompt(
            content_type=self.content_type,
            task=task,
            base_content=base_content,
            constraints=constraints
        )
        
        for attempt in range(1, self.retry_attempts + 1):
            try:
                if isinstance(self.llm_client, OpenAIClient):
                    response = self.llm_client.chat_completion(
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        model=self.model,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    content = response.choices[0].message.content.strip()
                    tokens = response.usage.total_tokens
                elif isinstance(self.llm_client, AnthropicClient):
                    response = self.llm_client.create_message(
                        messages=[{"role": "user", "content": prompt}],
                        model=self.model,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature
                    )
                    content = response.content[0].text
                    tokens = response.usage.input_tokens + response.usage.output_tokens
                else:
                    raise ValueError("Unknown client type")
                
                if log:
                    self.logger.info(f"Total tokens used: {tokens}")
                
                return content, tokens
                
            except Exception as e:
                self.logger.warning(
                    f"Attempt {attempt}/{self.retry_attempts} failed: {e}"
                )
                if attempt < self.retry_attempts:
                    sleep(self.delay)
                else:
                    raise
        
        raise RuntimeError("Failed to fit content to constraints")
    
    def fit_batch(
        self,
        constraints_df: pd.DataFrame,
        base_df: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fit base content to constraints for a batch of samples.
        
        Args:
            constraints_df: DataFrame with constraints (from constraints.csv)
            base_df: DataFrame with base content (from base_generated.csv)
            output_path: Optional path to save results
            
        Returns:
            DataFrame with fitted content
        """
        # Merge dataframes on instruction_number
        if "instruction_number" not in constraints_df.columns:
            raise ValueError("constraints_df must have 'instruction_number' column")
        if "instruction_number" not in base_df.columns:
            raise ValueError("base_df must have 'instruction_number' column")
        
        merged = pd.merge(
            constraints_df,
            base_df,
            on="instruction_number",
            suffixes=("_constraint", "_base")
        )
        
        self.logger.info(f"Fitting content for {len(merged)} samples")
        
        results = []
        for idx, row in merged.iterrows():
            instruction_num = row["instruction_number"]
            
            # Extract fields (handle suffix conflicts)
            task = row.get("main_task_constraint", row.get("main_task", ""))
            constraints = row["constraints"]
            base_content = row["base_content"]
            
            self.logger.info(f"Processing sample #{instruction_num}")
            
            try:
                fitted_content, tokens = self.fit_content(
                    task=task,
                    base_content=base_content,
                    constraints=constraints,
                    log=True
                )
                
                # Count constraints
                import re
                num_constraints = len(re.findall(r'^\d+\.', constraints, re.MULTILINE))
                
                results.append({
                    "instruction_number": instruction_num,
                    "main_task": task,
                    "constraints": constraints,
                    "base_content": base_content,
                    "fitted_content": fitted_content,
                    "fitted_length": len(fitted_content),
                    "num_constraints": num_constraints,
                    "model_used": self.model,
                    "tokens_used": tokens,
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                self.logger.error(
                    f"Failed to fit content for sample {instruction_num}: {e}"
                )
                results.append({
                    "instruction_number": instruction_num,
                    "main_task": task,
                    "constraints": constraints,
                    "base_content": base_content,
                    "fitted_content": "",
                    "fitted_length": 0,
                    "num_constraints": 0,
                    "model_used": self.model,
                    "tokens_used": 0,
                    "timestamp": datetime.now().isoformat()
                })
        
        result_df = pd.DataFrame(results)
        
        if output_path:
            result_df.to_csv(output_path, index=False, encoding="utf-8")
            self.logger.info(f"Fitted content saved to {output_path}")
        
        return result_df
