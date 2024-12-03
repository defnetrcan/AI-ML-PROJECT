# AI ML Project-Euphoria Dataset 
## Team Members:

-Defne Turcan

-Deniz Yakici

-Mete Alper Yegengil 

# Introduction
The aim of our project is to determine the performance of three machine learning models, Linear Regression, Random Forest, and Decision Tree, in predicting the **Happiness Index** using a dataset indicators. With this project we aimed to evaluate and find the best model for this task by analyzing the predictive accuracy of the models using metrics such as Mean Squared Error (MSE) and R² (coefficient of determination). The key objective of our project is to conclude a preprocessing, conducting Exploratory Data Analysis (EDA), and eventually do a model comparison to find the best model that would result in optimum performance for the prediction of happiness, considering accuracy, robustness, and generalization.

# Methods

## Design Choices

### Model Choice
As of requirements, in this project we selected three models specifically used for regression: Linear Regression, Random Forest, and Decision Trees. These models are used to predict our target variable which is happines_index. Our tasks were directly relevant to regression. This is because, first of all, our target variable "happiness_index" is numerical and continuous. Then, it can be seen that our task involves predicting the happiness_index by using other variables and features in our dataset. Therefore, we decided to test models that are applicable for regression. We chose 3 models as required to train: Linear Regression, Random Forest, Decision Trees. Linear Regression served as a simple baseline for comparison since it is simple and very interpretable, Random Forest is chosen to handle also the nonlinear relationships and outliers, and lastly, we chose decision trees to offer us more interpretability and a quicker evaluation.

### Other Relevant Design Decisions
-Evaluation Metrics
We selected **Mean Squared Error** to have an insight on the model accuracy. In addition, we chose **R²** which is the coefficient of determination to indicate us the variance.
-Hyperparameters
For Random Forest Model, we used GridSearchCV to tune n_estimators, max_depth, min_samples_split. For the Decision Tree Model, we only controlled max_depth to prevent overfitting.

## Describing the Dataset 
Our datasaet includes several so called socio-economic indicatures. 
The features of our dataset are:
- GDP per capita
- Social support
- Healthy life expectancy
- Freedom to make life choices
- Generosity
- Perception of corruption
The target variable of our dataset is **happiness_index**
## Python Environment 
In our project design we used multiple libraries:
-pandas
-numpy
-matplotlib
-seaborn
-scikit-learn (used for implementing machine learning models)


# Workflow
-1 Data Processing
-2 Model Training
-3 Model Evaluation

## Data Processing
To implement the machine learning models, we needed our data to be cleaned and appropriate for handling. We cleaned the data by getting rid of missing values, and we encoded categorical variables. Then, most importantly, we did feature scaling by using StandardScaler for Linear Regression since we needed our dataset values to be in the same range. Lastly, we did the train-test data split to evaluate models' performance and assure that our model is not affected by biased trained data.

## Model Training
We concluded hyperparameter tuning for Random Forest and Decision Tree Models by using cross-validation. The reason why we used cross-validation is to ensure generalizability of the models' performance, optimize hyperparameters, and decrease the risks of overfitting and underfitting.

## Model Evaluation
After running our code and trying it on the test set. We evaluated the model performance by using MSE and R². Eventually, we compared all od the results and by commenting and doing critical thinking we chose the best model.





