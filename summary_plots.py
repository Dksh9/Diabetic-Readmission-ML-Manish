import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Setup Paths
base_path = r'C:\Users\dkshp\OneDrive\Desktop\ML-Models-Manish'
csv_path = os.path.join(base_path, 'overall_results.csv')
output_path = os.path.join(base_path, 'model_comparison_leaderboard.png')

def create_comparison_visualization():
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please run your model cells first.")
        return

    # 2. Load the scoreboard data
    df = pd.read_csv(csv_path)
    
    # 3. Data Cleaning and Sorting
    # We sort by Recall_C1 to highlight the clinical priority of the project
    df = df.sort_values(by='Recall_C1', ascending=False)

    # 4. Initialize Plot
    # Increased figure height to 10 to accommodate the triple-bar clusters comfortably
    plt.figure(figsize=(15, 10))
    sns.set_style("whitegrid")
    
    # 5. Create grouped bar chart for Accuracy, Recall (Class 1), and F1-Score
    # Melt the dataframe to transform metrics into a single 'Score' column for seaborn
    df_plot = df.melt(id_vars='Model', 
                      value_vars=['Accuracy', 'Recall_C1', 'F1_Score'], 
                      var_name='Metric', 
                      value_name='Score')
    
    # Using 'viridis' palette for high contrast between the three metrics
    ax = sns.barplot(data=df_plot, x='Score', y='Model', hue='Metric', palette='viridis')

    # 6. Aesthetics and Labels
    plt.title('Final Model Performance Comparison: Diabetic Readmission (130-US Hospitals)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Score (0.0 - 1.0)', fontsize=12, fontweight='bold')
    plt.ylabel('Machine Learning Models', fontsize=12, fontweight='bold')
    plt.xlim(0, 1.1)  # Extended to 1.1 to give room for text labels
    plt.legend(title='Performance Metrics', loc='lower right', frameon=True, shadow=True)
    
    # Add precise numerical text labels on the bars
    for p in ax.patches:
        width = p.get_width()
        if width > 0:
            ax.text(width + 0.01, p.get_y() + p.get_height()/2, 
                    f'{width:.3f}', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    
    # 7. Save the final high-resolution image
    plt.savefig(output_path, dpi=300)
    plt.show()
    print(f"✅ Final comparison image with Accuracy, Recall, and F1-Score saved to: {output_path}")

if __name__ == "__main__":
    create_comparison_visualization()