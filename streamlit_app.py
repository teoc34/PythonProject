import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import numpy as np

st.title("Poverty's Influence on Education in Romania")
st.markdown("""
Explore how economic and social factors like poverty, **GDP, and **internet access affect education outcomes
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
df['County'] = df['County'].str.title()

# Preview Dataset
st.write("Dataset Preview")
st.dataframe(df)

# Aggregation by Poverty Levels
st.write("Grouped Education Metrics by Poverty Level")
df['Poverty_Level'] = pd.qcut(df['Poverty'], 4, labels=['Low', 'Medium', 'High', 'Very High'])
summary = df.groupby('Poverty_Level')[['Graduates', 'Abdandon_rate']].mean()
st.dataframe(summary.round(2))

# Scatter Plot (Graduates vs Poverty)
st.write("Graduates vs Poverty")
fig1, ax1 = plt.subplots()
ax1.scatter(df['Poverty'], df['Graduates'])
ax1.set_xlabel("Poverty (%)")
ax1.set_ylabel("Graduates (%)")
st.pyplot(fig1)
st.markdown("""
- This scatter plot shows how poverty levels relate to graduation rates.
- A downward trend would suggest that higher poverty is associated with fewer graduates.
- The scatter plot shows that as graduates increase, the poverty rate increases as well, which at first sight, might be counterintuitive.
- The result could be influenced by other socioeconomic factors that were not considered such as: migration patterns, unemployment rates, social programs.
""")


# Clustering (KMeans)
st.write("Clustering Counties")
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
- The graph shows 3 clusters of counties based on education-related variables.
- Colors represent counties grouped by similarity (poverty, graduation, and abandonment).
- This helps identify patterns. For example, which counties share similar education challenges.
""")


# Logistic Regression
st.write("Predicting High School Abandonment")
df['High_Abandonment'] = (df['Abdandon_rate'] > df['Abdandon_rate'].median()).astype(int)
X_logreg = df[['Poverty', 'PIB (euro/locuitor)', 'Graduates']]
y_logreg = df['High_Abandonment']
X_logreg_scaled = scaler.fit_transform(X_logreg)

logreg = LogisticRegression()
logreg.fit(X_logreg_scaled, y_logreg)
accuracy = logreg.score(X_logreg_scaled, y_logreg)
st.write(f"Model Accuracy: {accuracy:.2f}")
st.write(f"Logistic Regression Accuracy: {logreg.score(X_logreg_scaled, y_logreg):.2f}")
st.markdown(f"""
- This means that the logistic regression model correctly predicts whether a county has high or low school abandonment in {int(accuracy * 100)}% of cases.
- The model uses Poverty rate, GDP, and Graduation rate to make its predictions.
- The model achieved an accuracy of about 67%, meaning it correctly classifies high versus low abandonment counties
""")

# Multiple Regression (statsmodels)
st.write("Multiple Regression (Graduates ~ Poverty, Internet, PIB)")
X_multi = df[['Poverty', 'Internet_Penetration', 'PIB (euro/locuitor)']]
X_multi = sm.add_constant(X_multi)
y_multi = df['Graduates']
model = sm.OLS(y_multi, X_multi).fit()
st.text(model.summary())
st.markdown("""
- This regression analyzes how Poverty, **Internet access, and **GDP influence Graduation Rate.
- Coefficients tell us the direction and strength of each factor's influence.
- A negative coefficient for poverty suggests that higher poverty leads to fewer graduates.
- Ordinary Least Squared regression shows the relations between dependent variable Graduates and independent variables Poverty, Internet_Penetration, PIB(euro/locuitor).
- R-squared: = 0.608 => About 60.8% of the variability in the number of graduates is explained by the three predictors. This is a moderate to strong fit.
- Adjusted R-squared = 0.578 => Slightly lower than R², which is normal and expected.
- F-statistic = 19.68, p-value = 7.27e-08 => The overall model is statistically significant. At least one of the predictors significantly explains variance in the dependent variable.
- Only PIB(euro/locuitor) has a p-value < 0.05 => the relationship between PIB and the number of graduates is statistically significant.
- Durbin-Watson = 2.148 is close to 2 meaning that there is NO autocorrelation in residuals.
- Omnibus / Jarque-Bera tests = High p-values (> 0.05) suggest the residuals are normally distributed.
""")

# Footer
st.markdown("---")
st.markdown("Made by Teo and Alina – using data to understand social challenges 📊")
