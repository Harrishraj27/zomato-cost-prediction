# zomato-cost-prediction
This project employs machine learning to predict dining costs for two people at restaurants listed in a Zomato dataset and to classify whether orders are placed online or offline.

This project aims to predict the cost for two people at restaurants listed in the Zomato dataset and classify whether the orders were made online or offline. The project includes Exploratory Data Analysis (EDA) and Machine Learning modeling.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project explores a Zomato dataset to predict restaurant costs for two people and classify the order type using various machine learning models. We compare models to determine the best performing one and deploy it as a FastAPI application.

## Dataset

The dataset used in this project is `zomato.csv`, which contains information about restaurants, including ratings, votes, cost, and more.

## Installation

To set up the project on your local machine, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/zomato-cost-prediction.git
   cd zomato-cost-prediction
   ```

2. Create a virtual environment and activate it:
```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
```

3. Install the required packages:
```bash
   pip install -r requirements.txt
```
## Usage
Exploratory Data Analysis and Modeling
The notebook notebooks/EDA_and_Modeling.ipynb contains the code for data cleaning, EDA, and model training. Open the notebook and execute the cells to see the step-by-step analysis.

Running the API
To run the FastAPI server, navigate to the app directory and run the following command:
```bash
uvicorn main:app --reload
```
## Results
After evaluating several machine learning models, the Random Forest Regressor emerged as the most effective for predicting the cost for two people dining at restaurants. This model was selected based on its performance in terms of accuracy and reliability compared to other models, such as Linear Regression and Decision Tree Regressor.

**Model Evaluation** 
The evaluation of the models was conducted using the Root Mean Square Error (RMSE), a commonly used metric for assessing the accuracy of regression models. The RMSE provides a measure of the differences between the predicted values and the actual values in the dataset. A lower RMSE indicates a model with better predictive performance.

- **Random Forest Regressor**: The RMSE for this model was the lowest among the models tested, indicating that it made the most accurate predictions. The model's ability to capture non-linear relationships and interactions between features contributed significantly to its superior performance.

- **Linear Regression**: This model had a higher RMSE, suggesting that it was less effective in capturing complex patterns within the data. Linear Regression assumes a linear relationship between the features and the target variable, which may not fully represent the intricacies of restaurant cost prediction.

- **Decision Tree Regressor**: While this model performed better than Linear Regression, it was still outperformed by the Random Forest Regressor. Decision Trees can capture non-linear relationships, but they are prone to overfitting, which can limit their generalizability.

Overall, the successful implementation of this model showcases the potential for machine learning to enhance decision-making processes within the restaurant industry, ultimately leading to improved customer satisfaction and operational efficiency.

## Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

### .gitignore

Create a `.gitignore` file to specify which files and directories should not be tracked by Git.

