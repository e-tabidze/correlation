import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


features = ['country', 'destination', 'traffic', 'events']
# Read the generated data
data = pd.read_csv('network_data.csv', header=None)

# Calculate correlation matrix using Pearson formula
correlation_matrix = np.zeros((len(features), len(features)))
for i in range(len(features)):
    for j in range(len(features)):
        correlation_matrix[i, j] = pearsonr(data[i], data[j])[0]

# Plot correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', xticklabels=features, yticklabels=features)
plt.title('Correlation Matrix')
plt.xlabel('Features')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('correlation_matrix.pdf')
plt.close()

# Find features with highest correlation
max_corr = np.max(np.abs(correlation_matrix - np.eye(len(features))))
indices = np.where(np.abs(correlation_matrix - np.eye(len(features))) == max_corr)
feature1 = features[indices[0][0]]
feature2 = features[indices[1][0]]

# Extract data points of the two features with highest correlation
feature1_data = data[indices[0][0]]
feature2_data = data[indices[1][0]]

# Generate scatter diagram
plt.figure(figsize=(6, 4))
plt.scatter(feature1_data, feature2_data, color='blue')
plt.xlabel(feature1)
plt.ylabel(feature2)
plt.title('Scatter Diagram of Highest Correlation')
plt.savefig('highest_correlation.pdf')
plt.close()

print("Files generated successfully.")

