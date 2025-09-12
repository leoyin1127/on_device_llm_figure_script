import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Read the OSS Benchmarking Results CSV
df = pd.read_csv("OSS Benchmarking Results - Ophthalmology.csv")

# Extract the last row which contains the performance percentages
performance_row = df.iloc[-1]

# Get column names (model names) and their corresponding performance values
model_columns = df.columns[5:]  # Skip the first 5 columns (ID, Topic, Type, Question, GT)
performance_data = []

for col in model_columns:
    perf_value = performance_row[col]
    if pd.notna(perf_value) and isinstance(perf_value, str) and '%' in perf_value:
        # Extract percentage value and convert to float
        perf_float = float(perf_value.replace('%', ''))
        performance_data.append({'model': col, 'performance': perf_float})

# Convert to DataFrame for easier manipulation
perf_df = pd.DataFrame(performance_data)

# Group models by base type and extract version info
def categorize_model(model_name):
    model_lower = model_name.lower()
    if 'gpt-5' in model_lower:
        return 'GPT-5', model_name.split()[-1] if 'v' in model_name else 'v1'
    elif 'chatgpt-4o' in model_lower or 'chatgpt4o' in model_lower:
        return 'ChatGPT-4o', model_name.split()[-1] if 'v' in model_name else 'v1'
    elif 'deepseek' in model_lower:
        return 'DeepSeek', model_name.split()[-1] if 'v' in model_name else 'v1'
    elif 'qwen3' in model_lower or 'qwen-3' in model_lower:
        return 'Qwen3', model_name.split()[-1] if 'v' in model_name else 'v1'
    elif 'oss-20b' in model_lower or 'oss20b' in model_lower:
        config = 'L' if '(L)' in model_name else 'M' if '(M)' in model_name else 'H'
        version = model_name.split()[-1] if 'v' in model_name else 'v1'
        return f'OSS-20B ({config})', version
    elif 'oss-120b' in model_lower or 'oss120b' in model_lower:
        config = 'L' if '(L)' in model_name else 'M' if '(M)' in model_name else 'H'
        version = model_name.split()[-1] if 'v' in model_name else 'v1'
        return f'OSS-120B ({config})', version
    else:
        return 'Other', 'v1'

perf_df[['model_group', 'version']] = perf_df['model'].apply(lambda x: pd.Series(categorize_model(x)))

# Calculate mean and std for each model group
group_stats = perf_df.groupby('model_group')['performance'].agg(['mean', 'std']).fillna(0)

# Sort groups for consistent ordering - group OSS models together
group_order = ['GPT-5', 'ChatGPT-4o', 'DeepSeek', 'Qwen3', 'OSS-20B (L)', 'OSS-20B (M)', 'OSS-20B (H)', 
               'OSS-120B (L)', 'OSS-120B (M)', 'OSS-120B (H)']
group_stats = group_stats.reindex([g for g in group_order if g in group_stats.index])

# Create color mapping by model family
color_map = {}
gpt_groups = [g for g in group_stats.index if 'GPT' in g or 'ChatGPT' in g]
deepseek_groups = [g for g in group_stats.index if 'DeepSeek' in g]
qwen_groups = [g for g in group_stats.index if 'Qwen' in g]
oss20b_groups = [g for g in group_stats.index if 'OSS-20B' in g]
oss120b_groups = [g for g in group_stats.index if 'OSS-120B' in g]

# Generate colors for each family
if gpt_groups:
    blues = plt.colormaps.get_cmap("Blues").resampled(len(gpt_groups) + 2)
    for i, group in enumerate(gpt_groups):
        color_map[group] = blues(i + 1)

if deepseek_groups:
    reds = plt.colormaps.get_cmap("Reds").resampled(len(deepseek_groups) + 2)
    for i, group in enumerate(deepseek_groups):
        color_map[group] = reds(i + 1)

if qwen_groups:
    greens = plt.colormaps.get_cmap("Greens").resampled(len(qwen_groups) + 2)
    for i, group in enumerate(qwen_groups):
        color_map[group] = greens(i + 1)

if oss20b_groups:
    purples = plt.colormaps.get_cmap("Purples").resampled(len(oss20b_groups) + 2)
    for i, group in enumerate(oss20b_groups):
        color_map[group] = purples(i + 2)

if oss120b_groups:
    oranges = plt.colormaps.get_cmap("Oranges").resampled(len(oss120b_groups) + 2)
    for i, group in enumerate(oss120b_groups):
        color_map[group] = oranges(i + 2)

# Create positions with tight spacing within OSS groups
group_names = list(group_stats.index)
x_positions = []
current_pos = 0

for i, group_name in enumerate(group_names):
    if i > 0:
        prev_group = group_names[i-1]
        
        # Tight spacing within OSS model variants (0.6 instead of 1.0)
        if ('OSS-20B' in prev_group and 'OSS-20B' in group_name) or \
           ('OSS-120B' in prev_group and 'OSS-120B' in group_name):
            current_pos += 0.6
        # Normal spacing for all other transitions
        else:
            current_pos += 1.0
    
    x_positions.append(current_pos)

x_positions = np.array(x_positions)

# Define different bar widths for visual grouping
bar_widths = []
for group in group_names:
    if 'OSS-20B' in group or 'OSS-120B' in group:
        bar_widths.append(0.6)  # Narrower bars for OSS models
    else:
        bar_widths.append(0.8)  # Standard width for other models

# Create bar plot
plt.figure(figsize=(16, 8))
bars = []
for i, (pos, width) in enumerate(zip(x_positions, bar_widths)):
    bar = plt.bar(pos, group_stats['mean'].iloc[i], yerr=group_stats['std'].iloc[i], 
                  capsize=5, color=color_map.get(group_names[i], 'gray'),
                  edgecolor='white', linewidth=1.2, alpha=0.85, width=width)
    bars.extend(bar)

# Customize x-axis labels - make them cleaner
clean_labels = []
for group in group_stats.index:
    if 'OSS-20B' in group:
        clean_labels.append(group.replace('OSS-20B (', '').replace(')', ''))
    elif 'OSS-120B' in group:
        clean_labels.append(group.replace('OSS-120B (', '').replace(')', ''))
    else:
        clean_labels.append(group)

plt.xticks(x_positions, clean_labels, rotation=0, ha='center')
plt.xlabel("LLMs", fontsize=12, labelpad=15)
plt.ylabel("Diagnostic Accuracy (%)", fontsize=12)
plt.grid(axis="y", alpha=0.3)
plt.ylim(50, 90)  # Adjusted range for ophthalmology data

# Add value labels on bars
for i, (pos, mean_val) in enumerate(zip(x_positions, group_stats['mean'])):
    plt.text(pos, mean_val + 0.5, f'{mean_val:.1f}%', ha='center', va='bottom', 
             fontweight='bold', fontsize=9)

# Add subtle group indicators using brackets for OSS models
oss20b_positions = [pos for pos, group in zip(x_positions, group_names) if 'OSS-20B' in group]
oss120b_positions = [pos for pos, group in zip(x_positions, group_names) if 'OSS-120B' in group]

if oss20b_positions:
    # Draw a bracket pointing upward for OSS-20B bars
    y_bracket = 48.5
    plt.plot([min(oss20b_positions)-0.3, max(oss20b_positions)+0.3], [y_bracket, y_bracket], 
             'k-', linewidth=1, alpha=0.7, clip_on=False)
    plt.plot([min(oss20b_positions)-0.3, min(oss20b_positions)-0.3], [y_bracket, y_bracket+0.3], 
             'k-', linewidth=1, alpha=0.7, clip_on=False)
    plt.plot([max(oss20b_positions)+0.3, max(oss20b_positions)+0.3], [y_bracket, y_bracket+0.3], 
             'k-', linewidth=1, alpha=0.7, clip_on=False)
    plt.text(np.mean(oss20b_positions), y_bracket-0.8, 'OSS-20B', ha='center', va='center',
             fontsize=10, color='black', clip_on=False)

if oss120b_positions:
    # Draw a bracket pointing upward for OSS-120B bars
    y_bracket = 48.5
    plt.plot([min(oss120b_positions)-0.3, max(oss120b_positions)+0.3], [y_bracket, y_bracket], 
             'k-', linewidth=1, alpha=0.7, clip_on=False)
    plt.plot([min(oss120b_positions)-0.3, min(oss120b_positions)-0.3], [y_bracket, y_bracket+0.3], 
             'k-', linewidth=1, alpha=0.7, clip_on=False)
    plt.plot([max(oss120b_positions)+0.3, max(oss120b_positions)+0.3], [y_bracket, y_bracket+0.3], 
             'k-', linewidth=1, alpha=0.7, clip_on=False)
    plt.text(np.mean(oss120b_positions), y_bracket-0.8, 'OSS-120B', ha='center', va='center',
             fontsize=10, color='black', clip_on=False)

plt.tight_layout()
# Adjust subplot to make room for brackets below (after tight_layout)
plt.subplots_adjust(bottom=0.18)
plt.savefig('ophthalmology_llm_results.png', dpi=300, bbox_inches='tight')
plt.show()
