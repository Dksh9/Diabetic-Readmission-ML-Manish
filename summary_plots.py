import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the results
try:
    df = pd.read_csv('overall_results.csv')
except FileNotFoundError:
    print("Error: overall_results.csv not found. Please ensure the file exists.")
    exit()

# 2. Setup visual style
sns.set_style("whitegrid")
fig, ax1 = plt.subplots(figsize=(12, 7))

# 3. Bar Chart: Performance Comparison
df_melted = df.melt(id_vars='Model', var_name='Metric', value_name='Score')
plot = sns.barplot(data=df_melted, x='Metric', y='Score', hue='Model', palette='magma', ax=ax1)

plt.title('Final Model Evaluation: Diabetic Readmission Prediction', fontsize=16, fontweight='bold', pad=20)
plt.ylim(0, 0.8)
plt.ylabel('Performance Score (0.0 - 1.0)', fontsize=12)
plt.legend(title='Machine Learning Models', bbox_to_anchor=(1.05, 1), loc='upper left')

# Add numeric labels on top of bars
for p in plot.patches:
    if p.get_height() > 0:
        plot.annotate(format(p.get_height(), '.2f'), 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'center', 
                       xytext = (0, 9), 
                       textcoords = 'offset points',
                       fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('model_comparison_leaderboard.png', dpi=300)
print("Success: 'model_comparison_leaderboard.png' has been created in your root folder.")