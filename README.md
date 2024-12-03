# AI ML Project-Euphoria Dataset 
## Team Members:

-Defne Turcan 297411

-Deniz Yakici 304391

-Mete Alper Yegengil 

# Introduction
The aim of our project is to determine the performance of three machine learning models, Linear Regression, Random Forest, and Decision Tree, in predicting the **Happiness Index** using a dataset indicators. With this project we aimed to evaluate and find the best model for this task by analyzing the predictive accuracy of the models using metrics such as Mean Squared Error (MSE) and R² (coefficient of determination). The key objective of our project is to conclude a preprocessing, conducting Exploratory Data Analysis (EDA), and eventually do a model comparison to find the best model that would result in optimum performance for the prediction of happiness, considering accuracy, robustness, and generalization.

# Methods

## Design Choices


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
1. Data Processing
2. Explanatory Data Analysis
3. Model Training
4. Hyperparameter Tuning
5. Model Evaluation
6. Result Interpretation

# Experimental Design
The purpose of the experiments was to identify the learning method (regression, classification, clustering, etc.) and testing 3 models to compare the performance of these machine learning models to predict **happiness_index** based on the features of our dataset that are mainly socio-economic factors. Our main objective was to use relevant metrics and conducting necessary implementations to evaluate each of our models' ability to gather relationships between the variables in the dataset and our target variable as minimizing errors and maximizing the prediction accuracy. 

### Data Processing
To implement the machine learning models, we needed our data to be cleaned and appropriate for handling. We cleaned the data by getting rid of missing values, and we encoded categorical variables. Then, most importantly, we did feature scaling by using StandardScaler for Linear Regression since we needed our dataset values to be in the same range. Lastly, we did the train-test data split to evaluate models' performance and assure that our model is not affected by biased trained data.

### EDA
We preferred to do Correlation Heatmap and Distribution Analysis to better understand our dataset and the relationships between our variables. By doing EDA, we analyzed the distribution of features, their ranges, and scales. Then, we looked into the distribution of our happiness_index to make sure it is aligned with the assumptions of the models we used. Lastlyü we identified correlations and patterns among features to understand how they behave within. 

### Hyperparameter Tuning
Hyperparameter Tuning: Explanation

The hyperparameter tuning for the **Random Forest** and **Decision Tree** models was conducted to further improve their performance through optimization. This step was essential as it helped in balancing the overfitting-underfitting. 

The following hyperparameters were tuned for the Random Forest model:

- **`n_estimators`**: This is the number of trees in the forest, and the values being tested were `[50, 100, 150, 200]`
- **`max_depth`**: The value for `max_depth` limited how deep each tree will be, and this is a technique for model complexity control against overfitting, with values to be tested of `[5, 10, 15, 20, None]`
- **`min_samples_split`**: The minimum number of samples required to split an internal node, and it will be tested for `[2, 5, 10]`
- **`min_samples_leaf`**: Minimum number of samples required to be at a leaf node, taken from `[1, 2, 4]`, also to avoid too deep trees.
- In the case of the Decision Tree model, important hyperparameters were `max_depth` and `min_samples_split`, tested with `[5, 10, 15, 20, None]` and `[2, 5, 10]`, respectively, in order to avoid over-complexity of the tree. The grid of hyperparameter values was then tested using **GridSearchCV**, which performs a search over specified parameter values for the best-performing configuration.
  
### Model Training
We concluded hyperparameter tuning for Random Forest and Decision Tree Models by using cross-validation. The reason why we used cross-validation is to ensure generalizability of the models' performance, optimize hyperparameters, and decrease the risks of overfitting and underfitting.

### Model Evaluation
After running our code and trying it on the test set. We evaluated the model performance by using MSE and R². Eventually, we compared all of the results and we chose the best model.


## Model Choices and Baseline(s)
As of requirements, in this project we selected three models specifically used for regression. It can be seen that our task involves predicting the happiness_index by using other variables and features in our dataset. Therefore, we decided to test models that are applicable for regression.

- **Linear Regression**
  
  Linear Regression served as a simple baseline for comparison since it is simple and very interpretable. It directly assumed a linear relationship between the feature variables and our target variable which made it more convenient to use as a baseline for comparison with the other two models.

**Method**: We trained a Linear Regression model on the training set, and evaluated the model using Mean Squared Error (MSE) and R² metrics on the validation and test datasets.
  
- **Random Forests**
  
  Random Forest is chosen to handle also the nonlinear relationships and outliers, and lastly, we chose decision trees to offer us more interpretability and a quicker evaluation.
  
**Method**: We trained a Random Forest model on the dataset, and used GridSearchCV to tune hyperparameters like n_estimators (number of trees), max_depth, and min_samples_split.

- **Decision Trees**
  
  Decision Trees with is capability to capture some feature splits handled non-linearity. It has computationally low cost, requires minimal data preperation, allows flexibility.

**Method**: We trained Decision Tree models with varying max_depth to find the balance between overfitting and underfitting.
  


## Evaluation Metric Choices
We selected **Mean Squared Error** to have an insight on the model accuracy. Since it is used to measure average squared difference between predicted and actual values, we deduced our evaluation results as lower the values of MSE, better the prediction. Also, it is known to penalize large errors which makes it sensitive to outliers.  In addition, we chose **R²** which is the coefficient of determination to indicate us the variance for our second metric. It's values range from 0 to 1, 0 being no explanatory, 1 being perfect prediction. 



# Results
![image](https://github.com/user-attachments/assets/ba7298e7-dcb2-4bcf-9414-223dd7c811db)

According to the "Validation MSEs" Results, we can say that Linear Regression performs moderately, can predict slightly efficiently, but it is terrible with handling nonlinear relationships. On the other hand, Random Forest shows a higher error, probably because of the default hyperparameters. Lastly, the highest validation error belongs to Decision Tree model, due to overfitting or lack of depth control.

According to "Test Set Results", Linear Regression indicates us a very poor performance on the test set. Since R² is negative, we can deduce that it performs worse than the mean of the happiness_index. Random Forest Model performs best among the models, it captures nonlinear relationships and have a better interaction. Lastly, Decision Tree Model is seen to perform better than Linear Regression, but yet still no better than Random Forest. With its R² score, we can say that it lacks the robustness of ensemble models, in this case it is Random Forest.

Finally, with the results of "Best Hyperparameters", for Random Forest Model, it indicates that with no restriction on tree depth (max_depth=None), the model can grow fully and capture detailed relationships On the other hand, for Decision Tree Model, restricted depth of 5 seems to prevent overfitting, but this comes with the cost of not fully capturing the complexity of the data.

As a result, we can deduce that Random Forest Model is the best model out of all three models. It has the lowest MSE (361634.48) and the highest R² (0.3086), making it the best-performing model on the test set. Also, it captures complex interactions between features, making it more suitable for the dataset compared to Linear Regression. Lastly, as an ensemble method, Random Forest is less prone to overfitting compared to a single Decision Tree, offering better generalization.

# Conclusion

## Key Take Aways
In this project, the performances of Linear Regression, Decision Tree, and Random Forest are compared for the prediction of the Happiness Index based on the socio-economic indicators. Indeed, by the results presented, the best model for the data at hand, giving the least MSE and with the best R² score, indicating capability of capture of non-linear relationships of variable interactions and robust to generalized to new data. Decision Tree, with the use of depth control in order to avoid overfitting, showed a moderate performance. The baseline model, Linear Regression, had the worst performance since it failed to model the non-linear relationships of the features in the data. This project also emphasizes the usage of ensemble methods such as Random Forest when it comes to datasets containing complex interactions between features.

## Unanswered Questions and Future Work
Despite the promising results, some questions remain unanswered. The project did not consider other advanced ensemble methods such as Gradient Boosting or XGBoost, which can provide even better predictive performance. Again, feature engineering consisted of scaling and cleaning; domain knowledge-based transformations or additional features may prove helpful toward model accuracy. Lastly, the given dataset is assumed to be complete and representative, but the analysis could be validated by larger and more diverse datasets in order to guarantee applicability.

Some possible directions for future work:

1. **Hyperparameter Optimization**: Extend tuning to include additional hyperparameters and advanced techniques like Bayesian optimization. 

2. **Feature Engineering**: Introduce new features, interaction terms, or feature selection techniques to improve model insights and accuracy. 

3. **Alternative Models**: Evaluate Gradient Boosting, XGBoost, or neural networks to compare against Random Forest.
4. **Explainability**: Employ techniques such as SHAP or LIME to better understand model predictions and which features are most influential.







