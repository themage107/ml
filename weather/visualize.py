import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('weather/source.csv')

correlation_matrix = data.corr()

plt.figure(figsize=(10, 8)) 
sns.heatmap(correlation_matrix, 
            annot=True,          
            fmt=".2f",          
            cmap="coolwarm",     
            square=True,         
            linewidths=0.5)      
plt.title('Temperature Feature Heatmap')  
plt.show()