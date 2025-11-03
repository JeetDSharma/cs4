"""
Constraint generation module - extracts constraints from content.
"""

import pandas as pd
import logging
from time import sleep
from typing import Optional, Callable
from datetime import datetime

from cs4.core.prompts import get_constraint_generation_prompt
from cs4.utils.llm_client import OpenAIClient, AnthropicClient
from cs4.config import Config


class ConstraintGenerator:
    """Generate constraints from existing content."""
    
    def __init__(
        self,
        llm_client: Optional[object] = None,
        model: str = "gpt-4.1-mini",
        retry_attempts: int = 3,
        delay: float = 1.0
    ):
        """
        Initialize constraint generator.
        
        Args:
            llm_client: LLM client (OpenAI or Anthropic)
            model: Model identifier
            retry_attempts: Number of retry attempts on failure
            delay: Delay in seconds between retries
        """
        self.llm_client = llm_client or OpenAIClient(log_usage=True)
        self.model = model
        self.retry_attempts = retry_attempts
        self.delay = delay
        self.system_prompt = get_constraint_generation_prompt()
        
        self.logger = logging.getLogger("CS4Generator")
    
    def generate_constraints_for_content(
        self,
        content: str,
        log: bool = True
    ) -> tuple[str, str, int]:
        """
        Generate constraints for a single piece of content.
        
        Args:
            content: Input blog/story/news content
            log: Whether to log token usage
            
        Returns:
            Tuple of (main_task, constraints, tokens_used)
        """
        user_input = f"Input - {content}\nOutput -"
        
        for attempt in range(1, self.retry_attempts + 1):
            try:
                if isinstance(self.llm_client, OpenAIClient):
                    response = self.llm_client.chat_completion(
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": user_input}
                        ],
                        model=self.model
                    )
                    response_text = response.choices[0].message.content.strip()
                    tokens = response.usage.total_tokens
                elif isinstance(self.llm_client, AnthropicClient):
                    response = self.llm_client.create_message(
                        messages=[{"role": "user", "content": user_input}],
                        model=self.model,
                        system=self.system_prompt
                    )
                    response_text = response.content[0].text
                    tokens = response.usage.input_tokens + response.usage.output_tokens
                else:
                    raise ValueError("Unknown client type")
                
                # Parse response to extract main task and constraints
                main_task, constraints = self._parse_response(response_text)
                
                if log:
                    self.logger.info(f"Total tokens used: {tokens}")
                
                return main_task, constraints, tokens
                
            except Exception as e:
                self.logger.warning(
                    f"Attempt {attempt}/{self.retry_attempts} failed: {e}"
                )
                if attempt < self.retry_attempts:
                    sleep(self.delay)
                else:
                    raise
        
        raise RuntimeError("Failed to generate constraints")
    
    def _parse_response(self, response_text: str) -> tuple[str, str]:
        """
        Parse LLM response to extract main task and constraints.
        
        Args:
            response_text: Raw LLM response
            
        Returns:
            Tuple of (main_task, constraints)
        """
        lines = response_text.strip().split('\n')
        main_task = ""
        constraints_lines = []
        
        in_constraints = False
        for line in lines:
            line = line.strip()
            if line.startswith("Main Task:"):
                main_task = line.replace("Main Task:", "").strip()
            elif line.startswith("Constraints:"):
                in_constraints = True
            elif in_constraints and line:
                constraints_lines.append(line)
        
        constraints = "\n".join(constraints_lines)
        return main_task, constraints
    
    def generate_constraints_batch(
        self,
        df: pd.DataFrame,
        content_column: str = "Merged Blog",
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate constraints for a batch of content samples.
        
        Args:
            df: Input DataFrame with content
            content_column: Name of column containing content
            output_path: Optional path to save results
            
        Returns:
            DataFrame with constraints
        """
        if content_column not in df.columns:
            raise ValueError(f"Column '{content_column}' not found in DataFrame")
        
        self.logger.info(f"Processing {len(df)} samples")
        
        results = []
        for idx, row in df.iterrows():
            content = row[content_column]
            instruction_num = idx + 1
            
            self.logger.info(f"Processing instruction #{instruction_num}")
            
            try:
                main_task, constraints, tokens = self.generate_constraints_for_content(
                    content, log=True
                )
                
                results.append({
                    "instruction_number": instruction_num,
                    "instruction": content,
                    "main_task": main_task,
                    "constraints": constraints,
                    "model_used": self.model,
                    "tokens_used": tokens,
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                self.logger.error(
                    f"Failed to generate constraints for instruction {instruction_num}: {e}"
                )
                results.append({
                    "instruction_number": instruction_num,
                    "instruction": content,
                    "main_task": "",
                    "constraints": "",
                    "model_used": self.model,
                    "tokens_used": 0,
                    "timestamp": datetime.now().isoformat()
                })
        
        result_df = pd.DataFrame(results)
        
        if output_path:
            result_df.to_csv(output_path, index=False, encoding="utf-8")
            self.logger.info(f"Constraints saved to {output_path}")
        
        return result_df


def generate_constraints(
    df: pd.DataFrame,
    chat_fn: Callable,
    system_prompt: str,
    model: str = "gpt-4.1-mini",
    output_path: str = "../data/constraints.csv",
    retry_attempts: int = 3,
    delay: float = 1.0
) -> pd.DataFrame:
    """
    Legacy interface for generating constraints (backwards compatibility).
    
    This function maintains the original interface from blog_generation_type2.ipynb
    for easy migration of existing notebooks.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with a 'Merged Blog' column.
        chat_fn (callable): Function for LLM chat.
        system_prompt (str): Context or few-shot prompt for the model.
        model (str): LLM model identifier.
        output_path (str): Output CSV file path.
        retry_attempts (int): Number of retries per failed generation.
        delay (float): Delay (seconds) between retries.
        
    Returns:
        pd.DataFrame: Results with constraints
    """
    if 'Merged Blog' not in df.columns:
        raise ValueError("Input DataFrame must contain a 'Merged Blog' column.")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    constraints_list = []
    for idx, row in df.iterrows():
        instruction = row['Merged Blog']
        instruction_num = idx + 1
        logging.info(f"Processing instruction #{instruction_num}")
        
        constraints_text = ""
        for attempt in range(1, retry_attempts + 1):
            try:
                response = chat_fn(
                    instruction,
                    model=model,
                    system_prompt=system_prompt,
                    log=True
                )
                constraints_text = response.choices[0].message.content.strip()
                break
            except Exception as e:
                logging.warning(
                    f"Attempt {attempt}/{retry_attempts} failed for "
                    f"instruction {instruction_num}: {e}"
                )
                if attempt < retry_attempts:
                    sleep(delay)
                else:
                    logging.error(
                        f"Failed to generate constraints for instruction "
                        f"{instruction_num} after {retry_attempts} attempts."
                    )
        
        constraints_list.append({
            "Instruction Number": instruction_num,
            "Instruction": instruction,
            "Constraints": constraints_text
        })
    
    result_df = pd.DataFrame(constraints_list)
    result_df.to_csv(output_path, index=False)
    logging.info(f"Constraints saved to {output_path}")
    
    return result_df
