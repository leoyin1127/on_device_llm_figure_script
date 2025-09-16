import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from pathlib import Path
import sys

# Allow importing sibling helper without packaging
sys.path.append(str(Path(__file__).parent))
from radar_style import make_base_figure, color_pair, legend_label

# Read the Eurorad CSV data
data_path = Path(__file__).parent / 'data' / 'OSS Benchmarking Results - Eurorad.csv'
df = pd.read_csv(data_path)

# Get model columns (excluding metadata columns)
metadata_cols = ['case_id', 'Section', 'OriginalDescription', 'PostDescription', 
                 'DifferentialDiagnosisList', 'FinalDiagnosis', 'DiseaseLeak']
model_cols = [col for col in df.columns if col not in metadata_cols]

# Calculate accuracy for each model by comparing predictions to FinalDiagnosis
accuracy_by_section = {}

# Filter out NaN sections
valid_sections = df['Section'].dropna().unique()

for section in valid_sections:
    section_df = df[df['Section'] == section]
    section_accuracy = {}
    
    for model_col in model_cols:
        # Calculate accuracy: how many predictions match the final diagnosis
        correct = (section_df[model_col] == section_df['FinalDiagnosis']).sum()
        total = len(section_df)
        accuracy = (correct / total) * 100 if total > 0 else 0
        section_accuracy[model_col] = accuracy
    
    accuracy_by_section[section] = section_accuracy

# Group models by base type and calculate average across versions
# Plot (a): All model types with only (H) for OSS models
model_groups = {
    'gpt-5': [],
    'gpt-o4-mini': [],
    'DeepSeek-R1-0528': [],
    'Qwen3-235B': [],
    'gpt-oss-20B (H)': [],
    'gpt-oss-120B (H)': []
}

# Categorize each model column based on exact naming patterns
for col in model_cols:
    if 'gpt-5' in col.lower():
        model_groups['gpt-5'].append(col)
    elif 'o4-mini' in col.lower():
        model_groups['gpt-o4-mini'].append(col)
    elif 'deepseek' in col.lower():
        model_groups['DeepSeek-R1-0528'].append(col)
    elif 'qwen3' in col.lower():
        model_groups['Qwen3-235B'].append(col)
    elif 'oss-20b (h)' in col.lower():
        model_groups['gpt-oss-20B (H)'].append(col)
    elif 'oss-120b (h)' in col.lower():
        model_groups['gpt-oss-120B (H)'].append(col)

# Calculate average accuracy for each model group across sections
avg_accuracy_by_group = {}
sections = list(accuracy_by_section.keys())

for group_name, model_list in model_groups.items():
    if model_list:  # Only process groups that have models
        group_accuracies = []
        for section in sections:
            section_accs = [accuracy_by_section[section][model] for model in model_list 
                          if model in accuracy_by_section[section]]
            if section_accs:
                group_accuracies.append(np.mean(section_accs))
            else:
                group_accuracies.append(0)
        avg_accuracy_by_group[group_name] = group_accuracies

theta_labels = sections
fig = make_base_figure(theta_labels, width=1900, height=1200)

legend_shown = set()
# Add traces for each model group
for group_name, accuracies in avg_accuracy_by_group.items():
    if accuracies and any(acc > 0 for acc in accuracies):  # Only plot groups with data
        r_values = accuracies + [accuracies[0]]
        theta_values = theta_labels + [theta_labels[0]]
        family = legend_label(group_name)
        line_color, fill_color = color_pair(family)
        dash = "dash" if family == "gpt-5" else "solid"
        name = family
        show = family not in legend_shown
        fig.add_trace(
            go.Scatterpolar(
                r=r_values,
                theta=theta_values,
                line=go.scatterpolar.Line(color=line_color, width=2.5, dash=dash),
                marker=go.scatterpolar.Marker(color=line_color, size=4),
                fill="toself",
                fillcolor=fill_color,
                name=name,
                legendgroup=family,
                showlegend=show,
            )
        )
        legend_shown.add(family)

# Update layout with styling from reference
fig.update_layout(title=None)

# Create output directory if it doesn't exist
output_dir = Path(__file__).parent / 'output'
output_dir.mkdir(parents=True, exist_ok=True)

# Save the figure
output_png = output_dir / "eurorad_radar_plot_a.png"
output_html = output_dir / "eurorad_radar_plot_a.html"
fig.write_image(str(output_png), scale=3, engine="kaleido")

# Also save as interactive HTML
fig.write_html(str(output_html))

# Print summary statistics
print("\n=== Model Performance Summary (Plot A) ===")
print("Models included: All types with OSS (H) configurations only")
print(f"Sections analyzed: {len(sections)}")
print(f"Total cases: {len(df)}")
print("\nAverage accuracy by model group across all sections:")

overall_avg = {}
for group_name, accuracies in avg_accuracy_by_group.items():
    if accuracies and any(acc > 0 for acc in accuracies):
        overall_avg[group_name] = np.mean(accuracies)

# Sort by performance
sorted_groups = sorted(overall_avg.items(), key=lambda x: x[1], reverse=True)
for rank, (group, avg_acc) in enumerate(sorted_groups, 1):
    print(f"{rank}. {group}: {avg_acc:.2f}%")

print(f"\nRadar plot saved to: {output_png}")
print(f"Interactive version saved to: {output_html}")
