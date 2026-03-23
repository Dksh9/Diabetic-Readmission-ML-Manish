import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Setup Path (using your existing directory)
base_path = r'C:\Users\dkshp\OneDrive\Desktop\ML-Models-Manish'
# Ensure this points to your cleaned or raw dataset file
data_path = os.path.join(base_path, 'diabetic_data.csv') 
output_path = os.path.join(base_path, 'class_distribution.png')

def generate_distribution_chart():
    # 2. Load Data
    df = pd.read_csv(data_path)
    
    # 3. Create the Plot
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    
    # Plotting the 'readmitted' column
    # Note: If your data is already preprocessed to 0 and 1, use those labels.
    ax = sns.countplot(x='readmitted', data=df, palette='viridis')
    
    # 4. Aesthetics
    plt.title('Target Variable Distribution (Readmission vs. Non-Readmission)', fontsize=14, fontweight='bold')
    plt.xlabel('Readmission Status (0 = No, 1 = Yes)', fontsize=12)
    plt.ylabel('Number of Patient Records', fontsize=12)
    
    # 5. Add percentage labels on top of bars
    total = len(df)
    for p in ax.patches:
        percentage = f'{100 * p.get_height() / total:.1f}%'
        x = p.get_x() + p.get_width() / 2 - 0.05
        y = p.get_height() + (total * 0.01)
        ax.annotate(percentage, (x, y), fontsize=11, fontweight='bold')

    plt.tight_layout()
    
    # 6. Save
    plt.savefig(output_path, dpi=300)
    plt.show()
    print(f"✅ Class distribution chart saved to: {output_path}")

if __name__ == "__main__":
    generate_distribution_chart()