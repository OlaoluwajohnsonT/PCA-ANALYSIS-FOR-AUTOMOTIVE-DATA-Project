#!/usr/bin/env python
# coding: utf-8

# # PCA ANALYSIS FOR AUTOMOTIVE DATA

# The dataset is a comprehensive compilation of automotive attributes, encompassing 201 entries and 29 columns. It offers a rich array of information, ranging from numerical details like wheel-base, engine size, and price to categorical data such as make, aspiration, and body style. This dataset serves as a valuable resource for exploring the intricate characteristics of automobiles.
# 
# Among the notable aspects of this dataset are the presence of missing values in the 'stroke' and 'horsepower-binned' columns. These missing values warrant careful consideration during data preprocessing to ensure the integrity of any future analyses. Additionally, this dataset covers a diverse range of car makes and models, making it a suitable candidate for various analytical tasks and predictive modeling.
# 
# As I delve further into this dataset, I look forward to unveiling insightful trends, patterns, and relationships among automotive attributes. This exploration will enable me to gain a deeper understanding of the automotive domain, potentially leading to valuable insights for decision-making processes and predictive modeling in the automotive industry.

# In[2]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import sklearn
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
sklearn.__version__


# In[3]:


data = pd.read_csv("C:\\Users\\TOJMARK LTD\\auto_clean.csv")
data.head()


# In[4]:


data.info()


# In[6]:


#Checking for missing value
data.isnull().sum()


# In[7]:


# Assuming your DataFrame is named 'df'
data.dropna(axis=0, inplace=True)

# 'axis=0' indicates dropping rows with NaN values
# 'inplace=True' I update the DataFrame in place, so  I don't lose its dimension


# In[8]:


# Data Visualization
plt.figure(figsize=(10, 6))
sns.histplot(data['price'], kde=True)
plt.xlabel('Price')
plt.title('Price Distribution')
plt.show()


# In[9]:


plt.figure(figsize=(10, 6))
sns.boxplot(x='make', y='price', data=data)
plt.xlabel('Car Make')
plt.ylabel('Price')
plt.title('Price Distribution by Car Make')
plt.xticks(rotation=45)
plt.show()


# In[10]:


# Categorical Variable Analysis
plt.figure(figsize=(10, 6))
sns.countplot(x='body-style', data=data)
plt.xlabel('Body Style')
plt.ylabel('Count')
plt.title('Distribution of Body Styles')
plt.xticks(rotation=45)
plt.show()


# In[11]:


# Outlier Detection
plt.figure(figsize=(10, 6))
sns.boxplot(data['horsepower'])
plt.xlabel('Horsepower')
plt.title('Horsepower Distribution with Outliers')
plt.show()


# In[13]:


# Grouping and Aggregating
average_price_by_make = data.groupby('make')['price'].mean()
average_price_by_make


# In[14]:


# Pairplot to check for the data relationship
sns.pairplot(data[['wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-size', 'price']])
plt.show()


# In[15]:


# aspiration vs. Target
plt.figure(figsize=(10, 6))
sns.boxplot(x='aspiration', y='price', data=data)
plt.xlabel('Aspiration')
plt.ylabel('Price')
plt.title('Price Distribution by Aspiration')
plt.show()


# In[16]:


# Distribution of city mpg Variables
plt.figure(figsize=(10, 6))
sns.histplot(data['city-mpg'], kde=True)
plt.xlabel('City MPG')
plt.title('City MPG Distribution')
plt.show()


# # DROPPING NON-NUMERIC COLUMNS

# In[17]:


# Drop non-numeric columns
data_numeric = data.drop(['make', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location',
                          'engine-type', 'num-of-cylinders', 'fuel-system', 'horsepower-binned', 'diesel', 'gas'], axis=1)


# # Principal Conponent Analysis

# In[18]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_numeric)

# Perform PCA and choose the number of components based on the explained variance ratio
pca = PCA()
data_pca = pca.fit_transform(data_scaled)

# Calculate explained variance ratio
explained_var_ratio = pca.explained_variance_ratio_

# Determine the number of components explaining 95% of the variance
cumulative_var_ratio = np.cumsum(explained_var_ratio)
n_components = np.argmax(cumulative_var_ratio >= 0.95) + 1

# Fit PCA with the selected number of components
pca = PCA(n_components=n_components)
data_pca = pca.fit_transform(data_scaled)

# Create DataFrame for PCA results
pca_df = pd.DataFrame(data_pca, columns=[f'PC{i+1}' for i in range(n_components)])

# Add back non-numeric columns for interpretation
pca_df['make'] = data['make']
pca_df['body-style'] = data['body-style']

# View the results
print(pca_df.head())


# # #Create Scoring Models
# In this step, I can use the PCA results (pca_df) to build predictive models or perform further analysis
# For example, I will use the PC1 and PC2 scores as input features for regression or clustering models

# In[19]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Split the data into features (principal components) and target (car prices)
X = pca_df.drop(['make', 'body-style'], axis=1)
y = data['price']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict car prices on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2}")


# The R-squared value represents the proportion of the variance in the car prices that is explained by the model. In this case, the high R-squared value of 0.938 indicates that approximately 93.8% of the variability in car prices is accounted for by the principal components used in the model. This suggests that the model is performing well in predicting car prices.

# In[ ]:




