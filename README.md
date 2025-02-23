# House_price_Predictions
This project predicts house prices based on various features such as the number of bedrooms, space, lot size, tax, and more. It uses regression techniques, including Linear Regression and Gradient Boosting, to achieve accurate predictions. The Gradient Boosting model outperformed the baseline model with an R² score of 0.865 and a Mean Squared Error (MSE) of 22.76
Project Highlights
 Data cleaning and preprocessing
 Exploratory Data Analysis (EDA)
 Outlier detection and removal
 Multiple regression model training
 Model evaluation with MSE and R²
 Prediction function for user input
 Saved trained model using joblib
 Visualization of key dataset features
 Tech Stack & Libraries
Language: Python
Libraries:
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
 Installation Guide
1 Clone the repository:

bash
Copy
Edit
git clone https://github.com/alamin-safadi/house-price-prediction.git
cd house-price-prediction
2 Create a virtual environment (optional but recommended):

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Project Structure
bash
Copy
Edit
house-price-prediction/
│
├── data/                    # Dataset files
│   └── house_prices.csv
│
├── notebooks/               # Jupyter notebooks
│   └── EDA_and_Modeling.ipynb
│
├── models/                  # Saved ML models
│   └── gradient_boosting_model.pkl
│
├── src/                     # Source code
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
│
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── .gitignore               # Git ignore file
Exploratory Data Analysis (EDA)
Key insights from EDA:

Correlation between features and target (Price).
Distribution plots for each feature.
Boxplots to detect and remove outliers.
Heatmaps for feature relationships.
🤖 Model Performance
Model	MSE	R² Score
Linear Regression	67.61	0.5996
Gradient Boosting	22.76	0.8652
Gradient Boosting showed superior performance with better accuracy and lower error rates.

Making Predictions
After training the model, predict house prices using custom input:

python
Copy
Edit
from joblib import load
import numpy as np

# Load saved model
model = load('models/gradient_boosting_model.pkl')

# Example input: [Bedroom, Space, Room, Lot, Tax, Bathroom, Garage, Condition]
input_features = np.array([[3, 1100, 7, 50, 1099, 1.5, 1, 0]])
predicted_price = model.predict(input_features)

print(f"Predicted House Price: {predicted_price[0]:.2f}")
How to Run the Project
bash
Copy
Edit
# Run the training script
python src/train.py

# Predict using the trained model
python src/predict.py
Future Work
Incorporate more advanced models like XGBoost or LightGBM.
Add a web interface using Flask or Streamlit for real-time predictions.
Perform hyperparameter tuning for further accuracy improvements.
Contributing
Contributions are welcome!

Fork the repository.
Create your branch (git checkout -b feature/your-feature).
Commit your changes (git commit -m ' Add your feature').
Push to the branch (git push origin feature/your-feature).
Create a Pull Request.
