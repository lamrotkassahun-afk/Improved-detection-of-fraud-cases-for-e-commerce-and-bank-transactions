import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
import os

# 1. Load the processed data for visualization
# Using the final merged data from Task 1
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '../data/processed/fraud_data_final.csv')

df = pd.read_csv(DATA_PATH)

# Initialize the Dash app
app = dash.Dash(__name__)

# 2. Prepare Data for Visuals
# Summary Statistics
total_trans = len(df)
fraud_cases = df[df['class'] == 1].shape[0]
fraud_rate = (fraud_cases / total_trans) * 100

# Fraud by Country
fraud_by_country = df[df['class'] == 1].groupby('country').size().reset_index(name='fraud_count')
fig_country = px.choropleth(
    fraud_by_country, 
    locations="country", 
    locationmode='country names',
    color="fraud_count",
    title="Global Fraud Distribution",
    color_continuous_scale=px.colors.sequential.Reds
)

# Fraud by Device (Velocity insight)
# Assuming 'device_id' was used to create a frequency feature in Task 1
fig_device = px.histogram(
    df, x="class", color="class", 
    title="Class Distribution (0: Legit, 1: Fraud)",
    labels={'class': 'Transaction Class'},
    barmode='group'
)

# 3. Dashboard Layout
app.layout = html.Div([
    html.H1("Adey Innovations Inc. - Fraud Detection Dashboard", style={'textAlign': 'center'}),
    
    # Summary Boxes
    html.Div([
        html.Div([
            html.H3(f"Total Transactions: {total_trans}"),
        ], style={'width': '30%', 'display': 'inline-block', 'textAlign': 'center'}),
        html.Div([
            html.H3(f"Total Fraud Cases: {fraud_cases}"),
        ], style={'width': '30%', 'display': 'inline-block', 'textAlign': 'center', 'color': 'red'}),
        html.Div([
            html.H3(f"Fraud Rate: {fraud_rate:.2f}%"),
        ], style={'width': '30%', 'display': 'inline-block', 'textAlign': 'center'}),
    ], style={'padding': '20px', 'backgroundColor': '#f9f9f9'}),

    # Charts
    html.Div([
        dcc.Graph(figure=fig_country, style={'width': '48%', 'display': 'inline-block'}),
        dcc.Graph(figure=fig_device, style={'width': '48%', 'display': 'inline-block'}),
    ])
])

if __name__ == '__main__':
    app.run_眼(debug=True, port=8050)