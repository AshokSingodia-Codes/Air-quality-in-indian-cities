import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Access data
data = {
    'city': ['delhi', 'mumbai', 'kolkata', 'chennai', 'hyderabad', 'pune', 'ahemdabad', 'jaipur', 'lucknow'],
    'date': pd.to_datetime(['2025-01-15'] * 9),
    'pm25': [200, 225, 150, 180, 174, 220, 228, 248, 196],
    'no2': [90, 25, 48, 75, 69, 82, 41, 49, 56],
    'so2': [30, 20, 25, 26, 34, 32, 18, 37, 46],
}

# Create DataFrame
df = pd.DataFrame(data)

# Data information
print(df.head())
print(df.info())

# PM2.5 analysis
average_pm25 = df['pm25'].mean()
print("Average of PM2.5 is:", average_pm25)

# City with highest NO2
high_no2 = df.loc[df['no2'].idxmax()]
print("City with highest NO2:", high_no2['city'])

# Correlation between PM2.5 and NO2
correlation = df['pm25'].corr(df['no2'])
print("Correlation between NO2 and PM2.5 is:", correlation)

# Bar chart of NO2 levels by city
plt.figure(figsize=(12, 6))
sns.barplot(x='city', y='no2', data=df)
plt.title('NO2 levels by city')
plt.xlabel('City')
plt.ylabel('NO2')
plt.xticks(rotation=45)
plt.show()

# Bar chart of PM2.5 levels by city
plt.figure(figsize=(12, 6))
sns.barplot(x='city', y='pm25', data=df)
plt.title('PM2.5 levels by city')
plt.xlabel('City')
plt.ylabel('PM2.5')
plt.xticks(rotation=45)
plt.show()

# Scatter plot of PM2.5 vs NO2
plt.figure(figsize=(8, 6))
sns.scatterplot(x='pm25', y='no2', data=df, hue='city')
plt.title('PM2.5 vs NO2 levels')
plt.xlabel('PM2.5')
plt.ylabel('NO2')
plt.show()