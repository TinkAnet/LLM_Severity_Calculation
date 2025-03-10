import os
import json
import re
import matplotlib.pyplot as plt
import numpy as np
import glob
from collections import defaultdict

# Set matplotlib style for more professional plots
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'

# Create graph directory if it doesn't exist
graph_dir = "graph_compare"  # Changed directory name
if not os.path.exists(graph_dir):
    os.makedirs(graph_dir)
    print(f"Created directory: {graph_dir}")

# Directories containing JSON files
output_dirs = ["output", "output_for_prompt"]
all_category_scores = {}

# Function to extract exactly one final severity score per entry
def extract_final_scores(data, exclude_zeros=False):
    if not isinstance(data, list):
        return []
    
    scores = []
    for entry in data:
        if isinstance(entry, dict) and "Final Severity Score" in entry:
            try:
                score = float(entry["Final Severity Score"])
                if exclude_zeros and score == 0:
                    continue
                scores.append(score)
            except (ValueError, TypeError):
                print(f"  Error: Invalid severity score: {entry['Final Severity Score']}")
    
    return scores

for dir_name in output_dirs:
    # Dictionary to store category and its severity scores for this directory
    category_scores = defaultdict(list)
    
    # Get all JSON files in the current directory
    json_files = glob.glob(os.path.join(dir_name, "*.json"))
    
    print(f"\nProcessing files in {dir_name}:")
    for json_file in json_files:
        # Adjust regex pattern to match both naming patterns
        match = re.search(r'reasoning_(?:prompt_)?(.+?)_SAP200\.json', os.path.basename(json_file))
        if match:
            category = match.group(1)
            
            try:
                with open(json_file, 'r') as f:
                    print(f"Processing {json_file}...")
                    data = json.load(f)
                    # Exclude zeros only for original dataset
                    exclude_zeros = (dir_name == "output")
                    scores = extract_final_scores(data, exclude_zeros)
                    category_scores[category] = scores
                    
                    # Print detailed statistics
                    print(f"\nStatistics for {category} in {dir_name}:")
                    print(f"  Number of samples: {len(scores)}")
                    print(f"  Mean: {np.mean(scores):.3f}")
                    print(f"  Median: {np.median(scores):.3f}")
                    print(f"  Min: {min(scores):.3f}")
                    print(f"  Max: {max(scores):.3f}")
                    
            except Exception as e:
                print(f"Error processing {json_file}: {str(e)}")
    
    all_category_scores[dir_name] = category_scores

# Create two summary scatter plots (one for each dataset)
for dir_name, scores_dict in all_category_scores.items():
    plt.figure(figsize=(12, 7))
    
    # Use different colors for different categories
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    # Plot points for each category
    for i, (category, scores) in enumerate(scores_dict.items()):
        if not scores:
            continue
            
        # Create x-values centered around category position with jitter
        x_pos = i + 1
        jitter = np.random.normal(0, 0.1, len(scores))
        x_values = np.full_like(jitter, x_pos) + jitter
        
        # Plot points for this category
        color_idx = i % len(colors)
        # Clean up category name for display
        display_category = category.replace('1_', '').capitalize()
        plt.scatter(x_values, scores, alpha=0.7, s=25, color=colors[color_idx], 
                   marker='s', label=f"{display_category}")

    plt.xlabel('Category')
    plt.ylabel('Severity Score')
    title = "Response" if dir_name == "output" else "Prompt"
    plt.title(f'Severity Scores Distribution - {title}')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Set x-ticks at category positions
    categories = list(scores_dict.keys())
    # Clean up category names for x-axis
    display_categories = [cat.replace('1_', '').capitalize() for cat in categories]
    plt.xticks(np.arange(1, len(categories) + 1), 
               display_categories, 
               rotation=45, 
               ha='right')

    # Add border around the plot
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.0)

    # Position legend to the right of the plot
    plt.legend(loc='center left', 
              bbox_to_anchor=(1.02, 0.5), 
              frameon=True, 
              framealpha=0.9, 
              edgecolor='lightgray')

    plt.tight_layout()
    
    # Save the plot
    plot_filename = 'severity_scores_all_original.png' if dir_name == "output" else 'severity_scores_all_prompt.png'
    plt.savefig(os.path.join(graph_dir, plot_filename), dpi=300, bbox_inches='tight')
    plt.close()

print(f"\nScatter plots saved to {graph_dir}/ directory")

# Create statistical comparison plots
stat_measures = ['mean', 'median', 'min', 'max']
colors = {'output': 'blue', 'output_for_prompt': 'green'}
labels = {'output': 'Original', 'output_for_prompt': 'Prompt'}

# Bar plot for all statistical measures
plt.figure(figsize=(15, 8))
x_positions = np.arange(len(all_category_scores['output'].keys()))
width = 0.35

for measure in stat_measures:
    plt.figure(figsize=(12, 6))
    
    for i, (dir_name, scores_dict) in enumerate(all_category_scores.items()):
        stats = []
        categories = []
        
        for category in scores_dict.keys():
            scores = scores_dict[category]
            if measure == 'mean':
                stat = np.mean(scores)
            elif measure == 'median':
                stat = np.median(scores)
            elif measure == 'min':
                stat = np.min(scores)
            else:  # max
                stat = np.max(scores)
            stats.append(stat)
            categories.append(category.replace('1_', ''))
        
        offset = -width/2 if dir_name == 'output' else width/2
        plt.bar(x_positions + offset, stats, width, 
               label=labels[dir_name], 
               color=colors[dir_name], 
               alpha=0.7)
        
        # Add value labels on top of bars
        for x, stat in zip(x_positions + offset, stats):
            plt.text(x, stat, f'{stat:.2f}', 
                    ha='center', va='bottom')
    
    plt.xlabel('Category')
    plt.ylabel(f'{measure.capitalize()} Severity Score')
    plt.title(f'{measure.capitalize()} Severity Scores Comparison')
    plt.xticks(x_positions, [cat.capitalize() for cat in categories], 
               rotation=45, ha='right')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(graph_dir, f'severity_scores_{measure}_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

# Box plot comparison
plt.figure(figsize=(15, 8))
positions = []
labels = []
data = []

for i, category in enumerate(all_category_scores['output'].keys()):
    for j, dir_name in enumerate(['output', 'output_for_prompt']):
        if category in all_category_scores[dir_name]:
            positions.append(i * 3 + j)
            clean_category = category.replace('1_', '')
            label = f"{clean_category.capitalize()}\n({'Original' if dir_name == 'output' else 'Prompt'})"
            labels.append(label)
            data.append(all_category_scores[dir_name][category])

plt.boxplot(data, positions=positions, labels=labels)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Severity Score')
plt.title('Distribution Comparison (Box Plot)')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(graph_dir, 'severity_scores_boxplot_comparison.png'), 
            dpi=300, bbox_inches='tight')
plt.close()

# Violin plot comparison
plt.figure(figsize=(15, 8))
positions = []
labels = []
data = []

for i, category in enumerate(all_category_scores['output'].keys()):
    for j, dir_name in enumerate(['output', 'output_for_prompt']):
        if category in all_category_scores[dir_name]:
            positions.append(i * 3 + j)
            clean_category = category.replace('1_', '')
            label = f"{clean_category.capitalize()}\n({'Original' if dir_name == 'output' else 'Prompt'})"
            labels.append(label)
            data.append(all_category_scores[dir_name][category])

plt.violinplot(data, positions=positions, showmeans=True)
plt.xticks(positions, labels, rotation=45, ha='right')
plt.ylabel('Severity Score')
plt.title('Distribution Comparison (Violin Plot)')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(graph_dir, 'severity_scores_violin_comparison.png'), 
            dpi=300, bbox_inches='tight')
plt.close()

print(f"\nAdditional comparison plots saved to {graph_dir}/ directory:")
print("- Individual statistical measure comparisons (mean, median, min, max)")
print("- Box plot comparison")
print("- Violin plot comparison")
