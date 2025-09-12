# Medical LLM Benchmarking - Data Visualization

Data visualization scripts for generating publication figures from medical LLM benchmark results.

## Usage

```bash
# Setup
uv sync

# Generate bar charts
uv run python chart2bar/calculate_accuracy.py
uv run python chart2bar/visualization_barplot.py

# Generate radar plots
uv run python chart2radar/eurorad_radar_plot_a.py  # Plot (a)
uv run python chart2radar/eurorad_radar_plot_b.py  # Plot (b) 
uv run python chart2radar/eurorad_radar_plot_c.py  # Plot (c)

# Generate violin plots
uv run python chart2violin/nmed_error_violin_plot.py
```

## Structure

- `chart2bar/` - Bar charts comparing model accuracy
- `chart2radar/` - Radar plots for multi-dimensional comparison  
- `chart2violin/` - Violin plots for error distribution analysis

Place CSV data files in respective `data/` directories. Output figures are saved to `output/` directories.