import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

# Title & Research Goal
st.title("Poverty's Influence on Education in Romania")
st.markdown("""
This dashboard explores the relationship between **poverty** and **education outcomes** across Romanian counties.
It aims to identify patterns and risk areas through statistics, clustering, and predictive modeling.
""")

# Load and clean dataset
df = pd.read_csv("Dataset_pachete.csv")
df.columns = df.columns.str.strip()
df['Population'] = df['Population'].str.replace('.', '', regex=False).astype(int)
df['PIB (euro/locuitor)'] = df['PIB (euro/locuitor)'].str.replace(',', '', regex=False).astype(int)

# Display data
st.write("### Raw Data")
st.dataframe(df)

# Scatter: Poverty vs Graduation
st.write("### Graduation vs Poverty Level")
fig1, ax1 = plt.subplots()
ax1.scatter(df['Poverty'], df['Graduates'])
ax1.set_xlabel("Poverty (%)")
ax1.set_ylabel("Graduates (%)")
ax1.set_title("Higher Poverty → Lower Graduation?")
st.pyplot(fig1)

# Scatter: Poverty vs Abandonment
st.write("### School Abandonment vs Poverty Level")
fig2, ax2 = plt.subplots()
ax2.scatter(df['Poverty'], df['Abdandon_rate'])
ax2.set_xlabel("Poverty (%)")
ax2.set_ylabel("Abandonment Rate (%)")
ax2.set_title("Does Poverty Correlate with School Dropouts?")
st.pyplot(fig2)

# Grouping by poverty quartile
st.write("### 📊 Education Outcomes by Poverty Levels")
df['Poverty_Level'] = pd.qcut(df['Poverty'], 4, labels=['Low', 'Moderate', 'High', 'Very High'])
summary = df.groupby('Poverty_Level')[['Graduates', 'Abdandon_rate']].mean().round(2)
st.dataframe(summary)

# Clustering: Education & Poverty Risk Zones
st.write("### 🔍 Clustering Counties by Risk Profile")
features = df[['Poverty', 'Graduates', 'Abdandon_rate']]
X_scaled = StandardScaler().fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters

fig3, ax3 = plt.subplots()
scatter = ax3.scatter(df['Graduates'], df['Abdandon_rate'], c=clusters)
ax3.set_xlabel("Graduates (%)")
ax3.set_ylabel("Abandonment Rate (%)")
ax3.set_title("Clustered Counties by Education Risk")
st.pyplot(fig3)

# Logistic Regression: Predicting High Abandonment Risk
st.write("### Predicting Education Risk with Logistic Regression")

# Create binary target
df['High_Abandonment'] = (df['Abdandon_rate'] > df['Abdandon_rate'].median()).astype(int)
X = df[['Poverty', 'PIB (euro/locuitor)', 'Graduates']]
y = df['High_Abandonment']
X_scaled = StandardScaler().fit_transform(X)

logreg = LogisticRegression()
logreg.fit(X_scaled, y)

accuracy = logreg.score(X_scaled, y)
st.write(f"Model Accuracy: **{accuracy:.2f}**")

# Display coefficients
st.write("### 📋 Logistic Regression Feature Importance")
coeffs = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': logreg.coef_[0].round(3)
})
st.dataframe(coeffs)

# Summary
st.markdown("---")
st.markdown("Made with ❤️ by Teo and Alina – exploring data to fight inequality.")

