# CS4: Constraint-Satisfaction for Creative Content

A modular framework for evaluating LLM creativity through structured constraint satisfaction across multiple domains (blogs, stories, news).

## Installation

### Prerequisites

- Python 3.10+
- LLM API key (OpenAI, Anthropic, etc.)
- Conda (recommended)

### Setup

```bash
# Clone and navigate to project
git clone <repository-url>
cd cs4

# Create conda environment
cd env
conda env create -f environment.yaml
conda activate cs4

# Configure API keys
cp .env.example .env
# Edit .env with your API keys:
# OPENAI_API_KEY=sk-your-openai-key
# CLAUDE_API_KEY=sk-ant-your-claude-key
```

## Quick Start

```bash
python scripts/generate_constraints.py --domain blog --input-path data/processed/merged_blogs.csv --output-path data/outputs/constraints.csv
python scripts/generate_base.py --domain blog --input-path data/outputs/constraints.csv --output-path data/outputs/base_generated.csv
python scripts/fit_constraints.py --domain blog --constraints-path data/outputs/constraints.csv --base-path data/outputs/base_generated.csv --output-path data/outputs/fitted_content.csv
python scripts/evaluate.py --domain blog --input-path data/outputs/fitted_content.csv --output-path data/outputs/evaluation_results.csv
```

## Pipeline Stages

### Data Preparation

Before running the main pipeline, prepare your data:

```bash
# Merge blog pairs
python scripts/merge_blogs.py --input-path data/processed/blog_pairs.csv --output-path data/processed/merged_blogs.csv
```

### Stage 1: Constraint Generation

Extracts 39 atomic constraints from sample content.

```bash
python scripts/generate_constraints.py \
    --domain blog \
    --input-path data/processed/merged_blogs.csv \
    --output-path data/outputs/constraints.csv \
    --model gpt-4.1-mini
```

### Stage 2: Base Generation

Generates content from task description without constraints.

```bash
python scripts/generate_base.py \
    --domain blog \
    --input-path data/outputs/constraints.csv \
    --output-path data/outputs/base_generated.csv \
    --model gpt-4.1-mini
```

### Stage 3: Constraint Fitting

Revises base content to satisfy all 39 constraints.

```bash
python scripts/fit_constraints.py \
    --domain blog \
    --constraints-path data/outputs/constraints.csv \
    --base-path data/outputs/base_generated.csv \
    --output-path data/outputs/fitted_content.csv \
    --model gpt-4.1-mini
```

### Stage 4: Evaluation

Evaluates constraint satisfaction with detailed analysis.

```bash
python scripts/evaluate.py \
    --domain blog \
    --input-path data/outputs/fitted_content.csv \
    --output-path data/outputs/evaluation_results.csv \
    --model gpt-4.1-mini
```

## Configuration

### Environment Variables

Set these in your `.env` file:

```bash
OPENAI_API_KEY=sk-your-openai-key
CLAUDE_API_KEY=sk-ant-your-claude-key
```

### Default Models

Configure models in `cs4/config.py`:

- `DEFAULT_MERGE_MODEL`: Blog merging (default: "gpt-4.1-mini")
- `DEFAULT_CONSTRAINT_MODEL`: Constraint generation (default: "gpt-4.1-mini")  
- `DEFAULT_BASE_GEN_MODEL`: Base content generation (default: "gpt-4.1-mini")
- `DEFAULT_FITTING_MODEL`: Constraint fitting (default: "gpt-4.1-mini")
- `DEFAULT_EVALUATION_MODEL`: Evaluation (default: "gpt-4.1-mini")

### Key Parameters

- `NUM_CONSTRAINTS`: Number of constraints to generate (default: 39)
- `SIMILAR_THRESHOLD`: Similarity threshold for blog pairing (default: 0.75)
- `MAX_RETRIES`: API call retry attempts (default: 3)


## Monitoring

### Check Progress

```bash
# Monitor logs
tail -f logs/constraint_generation.log

# Check API usage
python -c "from cs4.utils.llm_client import get_total_usage; print(get_total_usage())"
```

### Analyze Results

```python
import pandas as pd

# Load evaluation results
df = pd.read_csv("data/outputs/evaluation_results.csv")

# Summary statistics
print(f"Total samples: {len(df)}")
print(f"Average satisfaction: {df['satisfaction_rate'].mean():.2%}")
print(f"Satisfaction distribution:")
print(df['satisfaction_rate'].describe())
```

## Project Structure

```
cs4/
├── cs4/                    # Core package
│   ├── core/              # Pipeline implementations
│   ├── utils/             # Utilities (LLM clients, config)
│   └── schemas/           # Data validation schemas
├── scripts/               # Pipeline execution scripts
├── configs/              # Configuration files
├── notebooks/            # Data preparation notebooks
├── env/                  # Environment setup
└── data/                 # Data directories (auto-created)
    ├── raw/
    ├── processed/
    ├── embeddings/
    └── outputs/
```

## License

This project is licensed under the MIT License.
