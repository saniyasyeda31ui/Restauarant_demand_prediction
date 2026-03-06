import joblib
import pandas as pd

# load model
model = joblib.load("restaurant_demand_model.pkl")

print("Model expects columns:", model.feature_names_in_)

sample = pd.DataFrame({
    "day_of_week": ["Saturday"],
    "temperature": [32],
    "weather": ["Sunny"],
    "season": ["Summer"],
    "event": ["No_Event"],
    "is_weekend": [1],
    "holiday": [0],
    "menu_item": ["Biryani"],
    "price": [200],
    "promotion": [0]
})

# check datatypes before prediction
print(sample.dtypes)

prediction = model.predict(sample)

print("Predicted demand:", prediction)