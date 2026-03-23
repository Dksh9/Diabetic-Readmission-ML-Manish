import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Setup Path
base_path = r'C:\Users\dkshp\OneDrive\Desktop\ML-Models-Manish'
data_path = os.path.join(base_path, 'diabetic_data.csv')

def generate_section_4_4_figures():
    if not os.path.exists(data_path):
        print("Error: diabetic_data.csv not found.")
        return
        
    # Load and basic prep for EDA
    df = pd.read_csv(data_path)
    df['readmitted_binary'] = df['readmitted'].replace({'>30': 1, '<30': 1, 'NO': 0})
    
    sns.set_style("whitegrid")

    # --- Figure 4.4: Class Distribution ---
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(x='readmitted_binary', data=df, palette='viridis', hue='readmitted_binary', legend=False)
    plt.title('Figure 4.4: Class Distribution of Target Variable', fontweight='bold')
    plt.xlabel('Readmission Status (0 = No, 1 = Yes)')
    plt.ylabel('Patient Count')
    plt.savefig(os.path.join(base_path, 'fig4_4_class_distribution.png'), dpi=300)
    plt.close()

    # --- Figure 4.5: Distribution of Key Features (Time in Hospital) ---
    plt.figure(figsize=(7, 5))
    sns.histplot(df['time_in_hospital'], bins=14, kde=True, color='teal')
    plt.title('Figure 4.5: Distribution of Key Features (Hospital Stay)', fontweight='bold')
    plt.xlabel('Days in Hospital')
    plt.savefig(os.path.join(base_path, 'fig4_5_feature_dist.png'), dpi=300)
    plt.close()

    # --- Figure 4.6: Feature vs Target (Inpatient Visits vs Readmission) ---
    plt.figure(figsize=(7, 5))
    # We use a boxplot to show how readmitted patients usually have higher prior visits
    sns.boxplot(x='readmitted_binary', y='number_inpatient', data=df, palette='magma', hue='readmitted_binary', legend=False)
    plt.ylim(0, 8) # Focused view on the most common range
    plt.title('Figure 4.6: Feature vs Target Relationship', fontweight='bold')
    plt.xlabel('Readmission (0=No, 1=Yes)')
    plt.ylabel('Number of Prior Inpatient Visits')
    plt.savefig(os.path.join(base_path, 'fig4_6_feature_vs_target.png'), dpi=300)
    plt.close()

    # --- Figure 4.7: Correlation Heatmap ---
    plt.figure(figsize=(10, 8))
    # Selecting core numeric indicators for a clean heatmap
    eda_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 
                'num_medications', 'number_inpatient', 'number_emergency', 'readmitted_binary']
    corr_matrix = df[eda_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', fmt=".2f", linewidths=0.5)
    plt.title('Figure 4.7: Correlation Heatmap (Selected Features)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, 'fig4_7_heatmap.png'), dpi=300)
    plt.close()

    print("SUCCESS: Section 4.4 Figures (4.4, 4.5, 4.6, 4.7) generated.")

if __name__ == "__main__":
    generate_section_4_4_figures()