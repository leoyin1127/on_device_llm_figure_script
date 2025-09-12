# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a data visualization project for generating publication-quality figures from medical LLM benchmarking results. The project processes OSS benchmarking data and creates bar charts, radar plots, and violin plots to visualize model performance across different medical domains (Eurorad, Ophthalmology, NMED).

## Development Commands

### Environment Setup
```bash
# Install dependencies using UV package manager
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### Running Visualizations
```bash
# Use uv run to execute scripts with proper dependencies
# Generate bar charts with accuracy calculations
uv run python chart2bar/calculate_accuracy.py
uv run python chart2bar/visualization_barplot.py

# Generate radar plots
uv run python chart2radar/plot_radar.py
uv run python chart2radar/eurorad_radar_plot.py

# Generate violin plots for NMED data
uv run python chart2violin/nmed_error_violin_plot.py
```

## Architecture and Structure

### Data Flow
1. **Input**: CSV files in `chart2*/data/` directories containing benchmark results
2. **Processing**: Python scripts calculate accuracy metrics and prepare data
3. **Output**: PNG/HTML visualizations saved to `chart2*/output/` directories

### Key Components

- **chart2bar/**: Bar chart generation for comparing model performance
  - Processes Eurorad and Ophthalmology benchmark results
  - Calculates accuracy percentages and generates grouped bar plots

- **chart2radar/**: Radar/spider plot visualization using Plotly
  - Creates multi-dimensional performance comparisons

- **chart2violin/**: Violin plot generation for error distribution analysis
  - Focuses on NMED (Diagnosis and Treatment) benchmark data
  - Shows distribution of model performance

### Data Format
Input CSV files typically contain:
- Model names as column headers
- Benchmark questions/tasks as rows
- Performance metrics (percentages, scores) as values
- Last row often contains aggregated performance percentages

### Visualization Libraries
- **matplotlib**: Static plots (bar charts, violin plots)
- **plotly**: Interactive plots (radar charts)
- **pandas**: Data manipulation and CSV processing
- **numpy**: Numerical computations

## Model Naming Standards

### Standard Model Names
- **gpt-5**: GPT-5 models (columns: gpt-5-0807-M1, gpt-5-0807-M2, gpt-5-0807-M3)
- **gpt-o4-mini**: GPT-4o mini models (columns: o4-mini-M1, o4-mini-M2, o4-mini-M3)
- **DeepSeek-R1-0528**: DeepSeek models (columns: deepseek r1 0528 v1, v2, v3)
- **Qwen3-235B**: Qwen3 models (columns: qwen3-235b v1, v2, v3)
- **gpt-oss-20B**: OSS 20B models with configurations:
  - (L): Low configuration (columns: oss-20b (L) v1, v2, v3)
  - (M): Medium configuration (columns: oss-20b (M) v1, v2, v3)
  - (H): High configuration (columns: oss-20b (H) v1, v2, v3)
- **gpt-oss-120B**: OSS 120B models with configurations:
  - (L): Low configuration (columns: oss-120b (L) v1, v2, v3)
  - (M): Medium configuration (columns: oss-120b (M) v1, v2, v3)
  - (H): High configuration (columns: oss-120b (H) v1, v2, v3)

### Radar Plot Variations
- **Plot (a)**: All model types with only (H) configuration for OSS models
- **Plot (b)**: OSS-120B all configurations (L, M, H) + gpt-5 as reference
- **Plot (c)**: OSS-20B all configurations (L, M, H) + gpt-5 as reference