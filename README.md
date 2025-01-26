# Title: Data Collection and Wrangling
import requests
import pandas as pd

# Example SpaceX API URL
url = "https://api.spacexdata.com/v4/launches"
response = requests.get(url)
data = response.json()

# Normalize JSON data to create a DataFrame
df = pd.json_normalize(data)

# Save the data to CSV
df.to_csv('spacex_data.csv', index=False)

# Display first few rows of the data
df.head()
python
Copy
Edit
# Title: Exploratory Data Analysis (EDA) and Visualization
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('spacex_data.csv')

# EDA: Check missing values and data types
print(df.info())

# Visualizations: Count the number of launches per LaunchSite
sns.countplot(x='LaunchSite', data=df)
plt.title('Number of Launches per Site')
plt.xlabel('Launch Site')
plt.ylabel('Count')
plt.show()

# Scatter plot of FlightNumber vs PayloadMass
sns.scatterplot(x='FlightNumber', y='PayloadMass', data=df)
plt.title('FlightNumber vs PayloadMass')
plt.xlabel('Flight Number')
plt.ylabel('Payload Mass')
plt.show()
python
Copy
Edit
# Title: SQL Analysis (Using SQLite as an example)
import sqlite3
import pandas as pd

# Create a connection to the SQLite database
conn = sqlite3.connect('spacex_database.db')

# Query to get the number of launches per site
query = """
SELECT LaunchSite, COUNT(*) as launch_count
FROM spacex_table
GROUP BY LaunchSite
"""

# Execute the query and store the result in a DataFrame
df_sql = pd.read_sql_query(query, conn)
print(df_sql)
python
Copy
Edit
# Title: Predictive Analysis (Machine Learning Model)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv('spacex_data.csv')

# Features and target
X = df[['PayloadMass', 'FlightNumber', 'Orbit', 'LaunchSite']]
y = df['Class']  # Assuming 'Class' is the target variable

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy}")
python
Copy
Edit
# Title: Interactive Dashboard with Plotly Dash
import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

# Load data
df = pd.read_csv('spacex_data.csv')

# Initialize Dash app
app = dash.Dash()

# Create a scatter plot
fig = px.scatter(df, x='FlightNumber', y='PayloadMass', color='Class')

# Define layout
app.layout = html.Div([
    html.H1('SpaceX Launch Analysis'),
    dcc.Graph(figure=fig)
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
python
Copy
Edit
# Title: Interactive Map with Folium
import folium

# Create a map centered around a specific location (Example: Kennedy Space Center)
m = folium.Map(location=[28.5721, -80.6480], zoom_start=10)

# Add a marker for the launch site
folium.Marker([28.5721, -80.6480], popup="Kennedy Space Center").add_to(m)

# Save map to HTML file
m.save("spacex_map.html")
