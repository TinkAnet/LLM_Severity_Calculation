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
graph_dir = "graph_for_prompt"
if not os.path.exists(graph_dir):
    os.makedirs(graph_dir)
    print(f"Created directory: {graph_dir}")

# Directory containing JSON files
output_dir = "output_for_prompt"

# Get all JSON files in the output directory
json_files = glob.glob(os.path.join(output_dir, "*.json"))

# Dictionary to store category and its severity scores (as lists for multiple scores)
category_scores = defaultdict(list)

# Function to extract exactly one final severity score per entry
def extract_final_scores(data):
    if not isinstance(data, list):
        return []
    
    scores = []
    for entry in data:
        if isinstance(entry, dict) and "Final Severity Score" in entry:
            try:
                score = float(entry["Final Severity Score"])
                scores.append(score)  # Include all scores, even zeros
            except (ValueError, TypeError):
                print(f"  Error: Invalid severity score: {entry['Final Severity Score']}")
    
    return scores

# Process each JSON file
print(f"Found {len(json_files)} JSON files to process")
for json_file in json_files:
    # Extract category from filename using regex
    # Pattern matches "reasoning_prompt_CATEGORY_SAP200.json"
    match = re.search(r'reasoning_prompt_(.+?)_SAP200\.json', os.path.basename(json_file))
    if match:
        category = match.group(1)
        
        # Read the JSON file
        try:
            with open(json_file, 'r') as f:
                print(f"Processing {json_file}...")
                data = json.load(f)
                
                # Extract exactly one final severity score per entry
                category_scores[category] = extract_final_scores(data)
                
                print(f"  Category: {category}, Found {len(category_scores[category])} valid scores")
                print(f"  Verify count: This should be close to 200. Actual count: {len(category_scores[category])}")
                
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")

# Print the summary before plotting
print("\nExtracted category scores:")
for category, scores in category_scores.items():
    print(f"{category}: {len(scores)} scores, range: {min(scores) if scores else 'N/A'} - {max(scores) if scores else 'N/A'}")

# Check if we have any data to plot
if not any(category_scores.values()):
    print("No valid severity scores were found. Please check the JSON file format.")
    exit(1)

# Create a separate scatter plot for each category
for category, scores in category_scores.items():
    if not scores:
        print(f"Skipping {category} - no valid scores found")
        continue
        
    plt.figure(figsize=(8, 6))
    
    # Create x-values (indices) for the scatter plot
    x_values = np.arange(len(scores))
    
    # Add some jitter to x-values to prevent overlapping points
    jitter = np.random.normal(0, 0.1, len(scores))
    jittered_x = x_values + jitter
    
    # Create scatter plot with smaller square markers
    plt.scatter(jittered_x, scores, alpha=0.8, s=25, c='green', marker='s')
    
    plt.xlabel('Data Point Index')
    plt.ylabel('Severity Score')
    plt.title(f'Severity Scores for {category.capitalize()}')
    
    # Add a border around the plot
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.0)
    
    # Add statistics as text with a more professional look
    stats_text = (
        f"n = {len(scores)}\n"
        f"Mean: {np.mean(scores):.3f}\n"
        f"Median: {np.median(scores):.3f}\n"
        f"Min: {min(scores):.3f}\n"
        f"Max: {max(scores):.3f}"
    )
    plt.annotate(stats_text, xy=(0.75, 0.20), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.9, edgecolor='lightgray'))
    
    # Set y-axis limits to show all data with some padding
    y_min, y_max = min(scores), max(scores)
    padding = (y_max - y_min) * 0.1 if y_max != y_min else 0.1
    plt.ylim(y_min - padding, y_max + padding)
    
    plt.tight_layout()
    
    # Save the graph to the graph directory
    plt.savefig(os.path.join(graph_dir, f'severity_scores_{category}.png'), dpi=300)
    
    print(f"Created scatter plot for {category}")

# Create a summary scatter plot comparing all categories
plt.figure(figsize=(12, 7))  # Wider figure to accommodate the legend

# Prepare data for combined scatter plot
all_categories = list(category_scores.keys())
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

for i, (category, scores) in enumerate(category_scores.items()):
    if not scores:
        continue
        
    # Create x-values centered around category position with jitter
    x_pos = i + 1
    jitter = np.random.normal(0, 0.1, len(scores))
    x_values = np.full_like(jitter, x_pos) + jitter
    
    # Plot points for this category using small square markers
    color_idx = i % len(colors)
    plt.scatter(x_values, scores, alpha=0.7, s=25, color=colors[color_idx], marker='s', label=category.capitalize())

plt.xlabel('Category')
plt.ylabel('Severity Score')
#plt.title('Severity Scores Across All Categories')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(np.arange(1, len(all_categories) + 1), [cat.capitalize() for cat in all_categories], rotation=45, ha='right')

# Add a border around the plot
for spine in plt.gca().spines.values():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(1.0)

# Position legend to the right of the plot instead of on top of it
plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True, framealpha=0.9, edgecolor='lightgray')

plt.tight_layout()

# Save the summary scatter plot to the graph directory with extra space for the legend
plt.savefig(os.path.join(graph_dir, 'severity_scores_all_categories.png'), dpi=300, bbox_inches='tight')

print("Created summary scatter plot for all categories")
print(f"\nAll scatter plots saved to {graph_dir}/ directory")
