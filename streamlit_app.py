import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import numpy as np

# Title & Description
st.title("📚 Poverty's Influence on Education in Romania")
st.markdown("""
Explore how economic and social factors like **poverty**, **GDP**, and **internet access** affect education outcomes
across Romanian counties. Includes clustering, regression models, and interactive data visualizations.
""")

# Load main dataset
df = pd.read_csv("Dataset_pachete.csv")
df.columns = df.columns.str.strip()
df['Population'] = df['Population'].str.replace('.', '', regex=False).astype(int)
df['PIB (euro/locuitor)'] = df['PIB (euro/locuitor)'].str.replace(',', '', regex=False).astype(int)

# Handle missing and extreme values
df = df.dropna()
df = df[df['Poverty'] < 100]

# Add Internet Penetration (simulated)
internet_df = pd.DataFrame({
    'County_ID': df['County_ID'],
    'Internet_Penetration': np.random.uniform(60, 90, len(df)).round(1)
})
df = pd.merge(df, internet_df, on='County_ID')

# Standardize County Names
df['County'] = df['County'].str.strip().str.lower()

# Preview Dataset
st.write("### 🗂️ Dataset Preview")
st.dataframe(df.head())

# Aggregation by Poverty Levels
st.write("### 📊 Grouped Education Metrics by Poverty Level")
df['Poverty_Level'] = pd.qcut(df['Poverty'], 4, labels=['Low', 'Medium', 'High', 'Very High'])
summary = df.groupby('Poverty_Level')[['Graduates', 'Abdandon_rate']].mean()
st.dataframe(summary.round(2))

# Scatter Plot (Graduates vs Poverty)
st.write("### 🎓 Graduates vs Poverty")
fig1, ax1 = plt.subplots()
ax1.scatter(df['Poverty'], df['Graduates'])
ax1.set_xlabel("Poverty (%)")
ax1.set_ylabel("Graduates (%)")
st.pyplot(fig1)
st.markdown("""
- This scatter plot shows how **poverty levels** relate to **graduation rates**.
- A downward trend would suggest that **higher poverty is associated with fewer graduates**.
- Look for clustering in the bottom-right (high poverty, low graduates) — these are potentially most at-risk counties.
""")


# Clustering (KMeans)
st.write("### 🤖 Clustering Counties")
features = ['Poverty', 'PIB (euro/locuitor)', 'Graduates', 'Abdandon_rate']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

fig2, ax2 = plt.subplots()
scatter = ax2.scatter(df['Graduates'], df['Abdandon_rate'], c=df['Cluster'])
ax2.set_xlabel("Graduates")
ax2.set_ylabel("Abandonment Rate")
st.pyplot(fig2)
st.markdown("""
- The graph shows **3 clusters** of counties based on **education-related variables**.
- Colors represent counties grouped by similarity (poverty, graduation, and abandonment).
- This helps identify patterns — for example, which counties share similar education challenges.
""")


# Logistic Regression
st.write("### 📉 Predicting High School Abandonment")
df['High_Abandonment'] = (df['Abdandon_rate'] > df['Abdandon_rate'].median()).astype(int)
X_logreg = df[['Poverty', 'PIB (euro/locuitor)', 'Graduates']]
y_logreg = df['High_Abandonment']
X_logreg_scaled = scaler.fit_transform(X_logreg)

logreg = LogisticRegression()
logreg.fit(X_logreg_scaled, y_logreg)
accuracy = logreg.score(X_logreg_scaled, y_logreg)
st.write(f"**Model Accuracy:** {accuracy:.2f}")
st.write(f"Logistic Regression Accuracy: **{logreg.score(X_logreg_scaled, y_logreg):.2f}**")
st.markdown(f"""
- This means that the logistic regression model correctly predicts **whether a county has high or low school abandonment** in **{int(accuracy * 100)}%** of cases.
- The model uses **Poverty rate, GDP, and Graduation rate** to make its predictions.
- A score of 0.67 is decent — it's better than random, but could still be improved with more features or tuning.
""")

# Multiple Regression (statsmodels)
st.write("### 📈 Multiple Regression (Graduates ~ Poverty, Internet, PIB)")
X_multi = df[['Poverty', 'Internet_Penetration', 'PIB (euro/locuitor)']]
X_multi = sm.add_constant(X_multi)
y_multi = df['Graduates']
model = sm.OLS(y_multi, X_multi).fit()
st.text(model.summary())
st.markdown("""
- This regression analyzes how **Poverty**, **Internet access**, and **GDP** influence **Graduation Rate**.
- Coefficients tell us the **direction and strength** of each factor's influence.
- A negative coefficient for poverty suggests that **higher poverty leads to fewer graduates**.
""")


# Geopandas Map (Choropleth)
st.write("### 🗺️ Romania County Map - Poverty Visualization")

# Load shapefile and match counties
romania_gdf = gpd.read_file("data/romania_Romania_Country_Boundary.zip!/romania_Romania_Country_Boundary.shp")
romania_gdf['County'] = romania_gdf['NAME'].str.strip().str.lower()

# Merge shapefile with data
merged_gdf = romania_gdf.merge(df, on='County')

# Plot
fig_map, ax_map = plt.subplots(figsize=(10, 8))
merged_gdf.plot(column='Poverty', cmap='OrRd', linewidth=0.8, ax=ax_map, edgecolor='0.8', legend=True)
ax_map.set_title('Poverty Rate by County in Romania', fontdict={'fontsize': '15'})
ax_map.axis('off')
st.pyplot(fig_map)
st.markdown("""
- This map shows the **poverty rate by county** using real geographic boundaries.
- Darker red = **higher poverty**.
- You can visually identify regions that may need more educational support or funding.
""")


# Footer
st.markdown("---")
st.markdown("Made by **Teo** and **Alina** – using data to understand social challenges 📊")
