# Car Price Prediction Using Python

## Project Overview
This project predicts the selling price of used cars using Machine Learning techniques in Python. It analyzes different car features such as present price, kilometers driven, fuel type, transmission, seller type, and car age to estimate the expected selling price.

Two machine learning algorithms are implemented:
- Linear Regression
- Random Forest Regression

## Dataset
**Dataset:** Vehicle Dataset from CarDekho

**Source:** [Kaggle - Vehicle Dataset from CarDekho](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)

- Original records: 301
- Records after removing duplicates: 299
- Original features: 9

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- KaggleHub

## Project Workflow
1. Download and load the dataset.
2. Explore the dataset and understand its features.
3. Check and handle missing values and duplicate records.
4. Perform data preprocessing and feature encoding.
5. Create a new feature called Car_Age.
6. Split the dataset into training and testing sets.
7. Train Linear Regression and Random Forest models.
8. Evaluate model performance using MAE, MSE, RMSE, and R² Score.
9. Visualize actual and predicted prices.
10. Predict car prices using user-provided inputs.

## Model Performance

| Evaluation Metric | Linear Regression | Random Forest |
|---|---:|---:|
| MAE | 1.47 | 1.41 |
| MSE | 6.37 | 11.04 |
| RMSE | 2.52 | 3.32 |
| R² Score | 0.75 | 0.57 |

Linear Regression achieved the higher R² score and lower RMSE on the test dataset.

## Visualizations

### 1. Actual vs Predicted Car Prices
This scatter plot compares actual car prices with prices predicted by the Linear Regression model.

### 2. Model Performance Comparison
This bar chart compares the R² scores of Linear Regression and Random Forest Regression.

## Sample Prediction

**Sample Input:**
- Present Price: 5.59 Lakhs
- Kilometers Driven: 27,000
- Car Age: 