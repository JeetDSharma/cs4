"""
Blog merging module - merges blog pairs using LLM.
Refactored from scripts/merge_blogs.py to follow CS4 pattern.
"""

import pandas as pd
import logging
from time import sleep
from typing import Optional
from datetime import datetime

from cs4.utils.llm_client import OpenAIClient, AnthropicClient
from cs4.config import Config


MERGE_SYSTEM_PROMPT = """You are a professional editor. Merge the two blogs below into a single coherent blog post.

Requirements:
- The result should read like a natural, single-authored blog
- Maintain the key ideas from both blogs
- Create smooth transitions between topics
- Ensure consistent tone and style throughout
- The merged blog should be comprehensive and well-structured
- Do not mention that it's a merge or reference "Blog 1" or "Blog 2"

Output only the merged blog text, with no preamble or explanation."""


class BlogMerger:
    """Merge blog pairs into single coherent blogs using LLM."""
    
    def __init__(
        self,
        llm_client: Optional[object] = None,
        model: str = "gpt-4-mini",
        retry_attempts: int = 3,
        delay: float = 1.0
    ):
        """
        Initialize blog merger.
        
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
        self.system_prompt = MERGE_SYSTEM_PROMPT
        
        self.logger = logging.getLogger("CS4Generator")
    
    def merge_pair(
        self,
        blog1: str,
        blog2: str,
        log: bool = True
    ) -> tuple[str, int]:
        """
        Merge two blogs using LLM.
        
        Args:
            blog1: First blog text
            blog2: Second blog text
            log: Whether to log progress
            
        Returns:
            Tuple of (merged_text, tokens_used)
        """
        user_prompt = f"Merge the two blogs below.\n\nBlog 1:\n{blog1}\n\nBlog 2:\n{blog2}"
        
        for attempt in range(1, self.retry_attempts + 1):
            try:
                if isinstance(self.llm_client, OpenAIClient):
                    response = self.llm_client.chat_completion(
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        model=self.model,
                        temperature=0.7,
                        max_tokens=2048
                    )
                    merged_text = response.choices[0].message.content.strip()
                    tokens = response.usage.total_tokens
                elif isinstance(self.llm_client, AnthropicClient):
                    response = self.llm_client.create_message(
                        messages=[{"role": "user", "content": user_prompt}],
                        model=self.model,
                        system=self.system_prompt,
                        max_tokens=2048,
                        temperature=0.7
                    )
                    merged_text = response.content[0].text
                    tokens = response.usage.input_tokens + response.usage.output_tokens
                else:
                    raise ValueError("Unknown client type")
                
                if log:
                    self.logger.info(f"Merged successfully ({tokens} tokens)")
                
                return merged_text, tokens
                
            except Exception as e:
                if log:
                    self.logger.warning(f"Attempt {attempt}/{self.retry_attempts} failed: {e}")
                if attempt < self.retry_attempts:
                    sleep(self.delay)
                else:
                    raise
        
        raise RuntimeError("Failed to merge blogs")
    
    def merge_pairs(
        self,
        pairs_df: pd.DataFrame,
        output_path: Optional[str] = None,
        save_interval: int = 10
    ) -> pd.DataFrame:
        """
        Merge all blog pairs in a DataFrame.
        
        Args:
            pairs_df: DataFrame with blog_1_text and blog_2_text columns
            output_path: Path to save results (saves incrementally if provided)
            save_interval: Save progress every N pairs
            
        Returns:
            DataFrame with merged blogs and metadata
        """
        required_columns = ['blog_1_text', 'blog_2_text']
        for col in required_columns:
            if col not in pairs_df.columns:
                raise ValueError(f"Required column '{col}' not found in input DataFrame")
        
        self.logger.info(f"Merging {len(pairs_df)} blog pairs...")
        
        merged_data = []
        
        for idx, row in pairs_df.iterrows():
            blog1 = str(row['blog_1_text'])
            blog2 = str(row['blog_2_text'])
            
            self.logger.info(f"Merging pair {idx + 1}/{len(pairs_df)}")
            
            try:
                merged_text, tokens = self.merge_pair(
                    blog1=blog1,
                    blog2=blog2,
                    log=True
                )
                
                merged_data.append({
                    'Original Blog 1': blog1,
                    'Original Blog 2': blog2,
                    'Merged Blog': merged_text,
                    'Similarity': row.get('similarity', None),
                    'blog_1_id': row.get('blog_1_id', None),
                    'blog_2_id': row.get('blog_2_id', None),
                    'merged_length': len(merged_text),
                    'model_used': self.model,
                    'tokens_used': tokens,
                    'timestamp': datetime.now().isoformat()
                })
                
                self.logger.info(f"  Merged length: {len(merged_text)} characters")
                self.logger.info(f"  Tokens used: {tokens}")
                
            except Exception as e:
                self.logger.error(f"Failed to merge pair {idx}: {e}")
                merged_data.append({
                    'Original Blog 1': blog1,
                    'Original Blog 2': blog2,
                    'Merged Blog': "",
                    'Similarity': row.get('similarity', None),
                    'blog_1_id': row.get('blog_1_id', None),
                    'blog_2_id': row.get('blog_2_id', None),
                    'merged_length': 0,
                    'model_used': self.model,
                    'tokens_used': 0,
                    'timestamp': datetime.now().isoformat()
                })
            
            # Save incrementally
            if output_path and (idx + 1) % save_interval == 0:
                temp_df = pd.DataFrame(merged_data)
                temp_df.to_csv(output_path, index=False, encoding="utf-8")
                self.logger.info(f"  Progress saved to {output_path}")
        
        result_df = pd.DataFrame(merged_data)
        
        if output_path:
            result_df.to_csv(output_path, index=False, encoding="utf-8")
            self.logger.info(f"Final results saved to {output_path}")
        
        return result_df
