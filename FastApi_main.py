import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Load the trained model and LabelEncoder
model_filename = 'Zomato_model.pkl'
label_encoder_filename = 'label_encoder.pkl'

loaded_model = joblib.load(model_filename)
label_encoder = joblib.load(label_encoder_filename)

app = FastAPI()

# Pydantic model for input data validation
class RestaurantInput(BaseModel):
    online_order: str
    book_table: str
    Ratings: float
    votes: int
    location: str
    rest_type: str
    dish_liked: str
    cuisines: str
    Type: str

# Endpoint to make predictions
@app.post("/predict/")
def predict_cost_for_two(data: RestaurantInput):
    try:
        # Convert the input data to a dictionary
        input_dict = data.dict()

        # Encode the categorical features using the loaded LabelEncoder
        categorical_cols = ['location', 'rest_type', 'dish_liked', 'cuisines', 'Type', 'online_order', 'book_table']
        for col in categorical_cols:
            input_value = input_dict[col]
            # Check if the input_value is in the known categories of the LabelEncoder
            if input_value in label_encoder.classes_:
                input_dict[col] = label_encoder.transform([input_value])[0]
            else:
                # Handle unseen category (e.g., new location) by encoding it as a new integer
                input_dict[col] = len(label_encoder.classes_)
                # Add the new category to the known classes of the LabelEncoder
                label_encoder.classes_ = np.append(label_encoder.classes_, input_value)

        # Create a DataFrame from the input data
        input_df = pd.DataFrame([input_dict])

        # Make the prediction for the cost for two people using the loaded model
        predicted_cost_for_two = loaded_model.predict(input_df)

        # Return the prediction as JSON response
        return {"Predicted Cost for Two People": predicted_cost_for_two[0]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
