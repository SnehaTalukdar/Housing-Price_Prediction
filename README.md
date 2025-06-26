# Using Linear Regression to Predict Housing Prices

-> Cleaning a housing dataset and using Simple Linear Regression and Multiple Linear Regression to forecast home prices are the main goals of this project.


## Utilized Dataset

- Source:  https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction
- Numerous features, such as area, number of bedrooms, bathrooms, furnishing status, etc., are included in the dataset.


##  Data Cleaning Procedures

1. Standardized Column Names - All column names were changed to lowercase, and underscores were used in place of spaces.

2. Numeric to Categorical Encoding
   - Mapped: - `yes` → 1, `no} → 0, - `unfurnished` → 0, `semi-furnished} → 1, `furnished` → 2.

The cleaned data was saved as `cleaned_housing_data.csv`.


##  Basic Linear Regression: Price versus Area

- Predicted `price` using only the `area` column.
-> MAE (Mean Absolute Error) ,and,

-> MSE (Mean Squared Error) - Both of them are used for evaluation.

-> R2 Score
A regression line comparing forecasted and actual prices was plotted.


## Multiple Linear Regression 

To predict `price`, all features were used.
The model was trained and assessed using an 80/20 train-test split.
The coefficients and intercept for every feature are shown.
  Metrics for evaluation:
     MAE, MSE, and R2 Score

###  Extra Illustrations

Residuals Plot: To evaluate model accuracy and randomness.
The correlation between each of the dataset's numerical features is displayed in the :- Correlation Heatmap.



## Technologies Employed :

- Python

- Pandas

- NumPy

- Matplotlib

- Seaborn

- scikit-learn

- Visual Studio Code (VS Code)



## Author

Name : Sneha Talukdar

Department : B.Tech CSE (AI & ML)

Location : Kolkata, West Bengal, India.