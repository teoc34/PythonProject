import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import numpy as np

# Title
st.title("Poverty's Influence on Education in Romania")
st.markdown("Explore how economic and social factors affect education outcomes across Romanian counties.")

# Load main dataset
df = pd.read_csv("Dataset_pachete.csv")
df.columns = df.columns.str.strip()
df['Population'] = df['Population'].str.replace('.', '', regex=False).astype(int)
df['PIB (euro/locuitor)'] = df['PIB (euro/locuitor)'].str.replace(',', '', regex=False).astype(int)

# Handle missing/extreme values
df = df.dropna()
df = df[df['Poverty'] < 100]  # Drop extreme outliers if any

# Merge with fake Internet Penetration dataset
internet_df = pd.DataFrame({
    'County_ID': df['County_ID'],
    'Internet_Penetration': np.random.uniform(60, 90, len(df)).round(1)
})
df = pd.merge(df, internet_df, on='County_ID')

# Streamlit display
st.write("Dataset Preview")
st.dataframe(df.head())

# Aggregation
st.write("Grouped Stats by Poverty Level")
df['Poverty_Level'] = pd.qcut(df['Poverty'], 4, labels=['Low', 'Medium', 'High', 'Very High'])
summary = df.groupby('Poverty_Level')[['Graduates', 'Abdandon_rate']].mean()
st.dataframe(summary.round(2))

# Matplotlib visualization
st.write("Graduates vs Poverty")
fig1, ax1 = plt.subplots()
ax1.scatter(df['Poverty'], df['Graduates'])
ax1.set_xlabel("Poverty (%)")
ax1.set_ylabel("Graduates (%)")
st.pyplot(fig1)

# Scaling
features_to_scale = ['Poverty', 'PIB (euro/locuitor)', 'Graduates', 'Abdandon_rate']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features_to_scale])

# Clustering
st.write("Clustering Counties (KMeans)")
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
fig2, ax2 = plt.subplots()
ax2.scatter(df['Graduates'], df['Abdandon_rate'], c=df['Cluster'])
ax2.set_xlabel("Graduates")
ax2.set_ylabel("Abandonment Rate")
st.pyplot(fig2)

# Logistic Regression
st.write("Predicting High Abandonment Risk")
df['High_Abandonment'] = (df['Abdandon_rate'] > df['Abdandon_rate'].median()).astype(int)
X_logreg = df[['Poverty', 'PIB (euro/locuitor)', 'Graduates']]
y_logreg = df['High_Abandonment']
X_logreg_scaled = scaler.fit_transform(X_logreg)

logreg = LogisticRegression()
logreg.fit(X_logreg_scaled, y_logreg)
st.write(f"Logistic Regression Accuracy: **{logreg.score(X_logreg_scaled, y_logreg):.2f}**")

# Multiple Regression with statsmodels
st.write("Multiple Regression (statsmodels)")
X_multi = df[['Poverty', 'Internet_Penetration', 'PIB (euro/locuitor)']]
X_multi = sm.add_constant(X_multi)
y_multi = df['Graduates']
model = sm.OLS(y_multi, X_multi).fit()
st.text(model.summary())

st.markdown("---")
st.markdown("Made by **Teo** and **Alina** – using data to understand social challenges 📊")
