"""
CS4: Constraint-Satisfaction for Creative Content

A modular framework for evaluating LLM creativity through structured
constraint satisfaction across multiple domains (blogs, stories, news).
"""

# Expose key classes and functions at package level
from cs4.core.constraint_generator import ConstraintGenerator
from cs4.core.base_generator import BaseGenerator
from cs4.core.constraint_fitter import ConstraintFitter
from cs4.core.evaluator import ConstraintEvaluator
from cs4.core.blog_merger import BlogMerger
from cs4.utils.llm_client import OpenAIClient, AnthropicClient
from cs4.schemas import (
    RawBlogsSchema,
    MergedBlogsSchema,
    ConstraintsSchema,
    BaseGeneratedSchema,
    FittedContentSchema,
    EvaluationResultsSchema,
)

__all__ = [
    "ConstraintGenerator",
    "BaseGenerator",
    "ConstraintFitter",
    "ConstraintEvaluator",
    "BlogMerger",
    "OpenAIClient",
    "AnthropicClient",
    "RawBlogsSchema",
    "MergedBlogsSchema",
    "ConstraintsSchema",
    "BaseGeneratedSchema",
    "FittedContentSchema",
    "EvaluationResultsSchema",
]
