import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import re

def parse_nmed_csv(filename):
    """Parse NMED CSV file and extract scores properly"""
    model_errors = {}
    
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Get header to understand column positions
    header = lines[0].strip().split(',')
    model_columns = header[6:]  # Skip first 6 metadata columns
    
    # Initialize error lists for each model
    for model in model_columns:
        model_errors[model.strip()] = []
    
    for i, line in enumerate(lines[1:], 1):  # Skip header
        line = line.strip()
        if not line:
            continue
            
        # Split line by comma
        parts = line.split(',')
        
        # Check if this line contains scores (ends with numeric values)
        if len(parts) >= len(header):
            # Check if the last few parts contain numeric scores
            score_parts = parts[6:]  # Skip metadata columns
            numeric_count = 0
            scores = []
            
            for part in score_parts:
                part = part.strip().strip('"')
                if part and re.match(r'^[0-9]*\.?[0-9]+$', part):
                    numeric_count += 1
                    scores.append(float(part))
                elif part:
                    scores.append(None)
                else:
                    scores.append(None)
            
            # If we have enough numeric scores, this is likely a score line
            if numeric_count >= 5:  # At least 5 numeric scores
                # Get human eval score (should be in column 5, index 5)
                human_score_str = parts[5].strip().strip('"')
                
                if human_score_str and re.match(r'^[0-9]*\.?[0-9]+$', human_score_str):
                    human_score = float(human_score_str)
                    
                    # Calculate errors for each model
                    for j, (model_col, model_score) in enumerate(zip(model_columns, scores)):
                        if model_score is not None:
                            error = human_score - model_score
                            model_errors[model_col.strip()].append(error)
    
    return model_errors

def create_violin_plot(model_errors, title, filename):
    """Create violin plot for a single dataset"""
    # Create series and values lists following the reference format
    series = []
    values = []

    # Add errors for each model (following the reference pattern)
    for model, errors in model_errors.items():
        if errors:  # Only add models that have data
            values.extend(errors)
            series.extend([model] * len(errors))

    # Create DataFrame following the reference format
    dic = {"series": series, "values": values}
    df = pd.DataFrame(dic)

    # Create figure following the reference style
    fig = go.Figure()

    # Add violin plots for each category (following reference format)
    for category in df['series'].unique():
        fig.add_trace(go.Violin(x=df['series'][df['series'] == category],
                                y=df['values'][df['series'] == category]))

    # Add box plot overlay (following reference format)
    fig.add_trace(go.Box(x=df['series'],
                         y=df['values'],
                         width=0.1,
                         fillcolor="lightgray",
                         line=dict(width=0.8, color='black')))

    # Update layout following the reference format
    fig.update_layout(showlegend=False)
    fig.update_yaxes(range=[-4.1, 4.1])
    fig.update_layout(
        font=dict(family='times new roman', size=17, color="#000000"),
        width=1400,
        height=550,
        xaxis=dict(title=""),
        yaxis=dict(title="Error",
                   tickvals=[-4, -3, -2, -1, 0, 1, 2, 3, 4],
                   ticktext=["-4.00", "-3.00", "-2.00", "-1.00", "0.00", "1.00", "2.00", "3.00", "4.00"],
                   showticklabels=True),
        title=dict(text='')
    )

    fig.update_layout(margin=dict(l=0, r=0, b=0, t=25))

    # Save the plot following reference format
    fig.write_image(filename, width=1400, height=550, scale=1)
    
    return len(values), model_errors

# Process diagnosis data
print("Loading and parsing diagnosis data...")
diagnosis_model_errors = parse_nmed_csv("OSS Benchmarking Results - NMED Diagnosis.csv")

print("Creating diagnosis violin plot...")
diagnosis_count, diagnosis_errors = create_violin_plot(diagnosis_model_errors, "Diagnosis", "nmed_diagnosis_violin.png")

# Process treatment data  
print("Loading and parsing treatment data...")
treatment_model_errors = parse_nmed_csv("OSS Benchmarking Results - NMED Treatment.csv")

print("Creating treatment violin plot...")
treatment_count, treatment_errors = create_violin_plot(treatment_model_errors, "Treatment", "nmed_treatment_violin.png")

# Display summary statistics
print(f"\n=== DIAGNOSIS DATA ===")
print(f"Total error calculations: {diagnosis_count}")
for model, errors in diagnosis_errors.items():
    if errors:
        print(f"{model}: Count: {len(errors)}, Mean: {np.mean(errors):.3f}, Std: {np.std(errors):.3f}")

print(f"\n=== TREATMENT DATA ===")
print(f"Total error calculations: {treatment_count}")
for model, errors in treatment_errors.items():
    if errors:
        print(f"{model}: Count: {len(errors)}, Mean: {np.mean(errors):.3f}, Std: {np.std(errors):.3f}")

print(f"\nGenerated files:")
print(f"- nmed_diagnosis_violin.png")
print(f"- nmed_treatment_violin.png")

# Uncomment to show plot interactively
# fig.show()