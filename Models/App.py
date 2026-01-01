import joblib
import pandas as pd
import numpy as np
from dash import Dash, html, dcc, Input, Output

# --------------------------------------------------
# Load trained pipeline (same folder as App.py)
# --------------------------------------------------
model = joblib.load("xgb_price_model.pkl")

# --------------------------------------------------
# Create Dash app
# --------------------------------------------------
app = Dash(__name__)

# --------------------------------------------------
# Layout
# --------------------------------------------------
app.layout = html.Div([

    html.H1("Airbnb Price Prediction Dashboard"),

    # -------- Room type --------
    html.Label("Room Type"),
    dcc.Dropdown(
        id="room_type",
        options=[
            {"label": "Entire home / apartment", "value": "Entire home/apt"},
            {"label": "Private room", "value": "Private room"},
            {"label": "Shared room", "value": "Shared room"}
        ],
        value="Entire home/apt"
    ),

    html.Br(),

    # -------- Neighbourhood group --------
    html.Label("Neighbourhood Group"),
    dcc.Dropdown(
        id="neighbourhood_group",
        options=[
            {"label": "Manhattan", "value": "Manhattan"},
            {"label": "Brooklyn", "value": "Brooklyn"},
            {"label": "Queens", "value": "Queens"},
            {"label": "Bronx", "value": "Bronx"},
            {"label": "Staten Island", "value": "Staten Island"}
        ],
        value="Manhattan"
    ),

    html.Br(),

    # -------- Host type --------
    html.Label("Host Type"),
    dcc.Dropdown(
        id="host_type",
        options=[
            {"label": "Individual host", "value": "individual"},
            {"label": "Professional host / company", "value": "professional"}
        ],
        value="individual"
    ),

    html.Hr(),

    # ---- UPDATED LABEL ----
    html.H2("Predicted price per night (USD)"),
    html.H3(id="predicted_price")
])

# --------------------------------------------------
# Callback
# --------------------------------------------------
@app.callback(
    Output("predicted_price", "children"),
    Input("room_type", "value"),
    Input("neighbourhood_group", "value"),
    Input("host_type", "value"),
)
def predict_price(room_type, neighbourhood_group, host_type):

    # Map host type to numeric value
    host_listings_count = 1 if host_type == "individual" else 10

    # Full feature set expected by the pipeline
    X = pd.DataFrame([{
        "room_type": room_type,
        "neighbourhood_group": neighbourhood_group,
        "calculated_host_listings_count": host_listings_count,

        # ---- fixed internal features ----
        "minimum_nights": 3,
        "availability_365": 180,
        "neighbourhood": "Harlem",
        "latitude": 40.7831,
        "longitude": -73.9712,
        "number_of_reviews": 0,
        "reviews_per_month": 1.0,
        "last_review_year": 2019
    }])

    # Model predicts log(price) → convert back
    log_price = model.predict(X)[0]
    price = np.exp(log_price)

    return f"${price:.2f}"

# --------------------------------------------------
# Run app
# --------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
