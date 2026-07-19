import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('healthcare-dataset-stroke-data-selected-columns.csv')

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

sns.histplot(data=df, x='age', kde=True, color='royalblue', bins=20)

plt.title('Histogram of Release Year Distribution', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Release Year', fontsize=12)
plt.ylabel('Count / Frequency', fontsize=12)

plt.tight_layout()
plt.show()

corr =df.corr(numeric_only="True")
sns.heatmap(corr,annot=True,cmap="coolwarm")
plt.show()