import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv("house_price_bd.csv")
df.dropna(inplace=True)

# Keep only numeric columns for PCA
df_numeric = df.select_dtypes(include=['number'])

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_numeric)
print(scaled_data)

# Apply PCA (reduce to 2 components)
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# Create a DataFrame for PCA results
pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
print(df.columns)
# Add target variable back (if exists)
if 'Price_in_taka' in df.columns:
    pca_df['Price'] = df['Price_in_taka'].replace({'৳':'',',':''}, regex=True)

print(pca_df.head())
# pca_df['Price'] = pca_df['Price'].replace({'৳':'',',':''})
# print(pca_df.head())