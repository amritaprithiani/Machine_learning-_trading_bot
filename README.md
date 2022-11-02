# Machine_learning-_trading_bot

This repository is to create and tries to optimize a trading algorithmic  machine learning model to automate the trading decisions. first step is to create a base line model that is based on Support Vector Machines and then it is been optimize by adjusting input parameters, training size. Lastly, Appling new machine learning model to compare and evaluate better performing model.

-----

## Technologies

```python
Language: Python 3.9.12

Libraries used:

Pandas 
Jupyter Labs 
Pathlib 
Scikit-Learn
Matplot Lib
PyViz hvPlot
DateOffset

```

-----

## Installation Guide

```python
conda install pandas
conda install jupyterlab
conda install -U Scikit-Learn
conda install -c conda-forge matplotlib
conda install -c pyviz hvplot

Check the to make sure everything has been installed properly
conda list pandas
conda list jupyter lab
conda list Scikit-learn
conda list matplotlib
conda list hvplot

```

----

## Usage

To run this analysis jupyter lab notebook has been used. To run jupyter lab you need to use GitBash and navigate to where you have exported the files associated with this project and activate your dev environment. Next, this project can be ran by navigating to the crypto_investments.ipynb jupyter notebook file.

----

## Evaluation Report

## Establishing a Baseline Performance

Purpose - The purpose of baseline model(SVM) is to identify the technical signals to make trading decision. If actual returns on close price are more than 0 the model suggests to buy shares that is to go long strategy. Where as if the actual return is less than 0 it signals to go short sell. 

Input features - 4 days short-window SMA  and 100 dys long-window SMA

Traning Period - 3 months

Classification report - 

```python
            precision    recall  f1-score   support

        -1.0       0.43      0.04      0.07      1804
         1.0       0.56      0.96      0.71      2288

    accuracy                           0.55      4092
   macro avg       0.49      0.50      0.39      4092
weighted avg       0.50      0.55      0.43      4092
```

![cumulative return](/Starter_Code/cumulative_return_plot.png)

## Tunning the Baseline Trading Algorithm

### The baseline model is being modified by updating the input features

Short-window - 50 days
long-window -200 days

classification report -

```python
             precision    recall  f1-score   support

        -1.0       0.45      0.20      0.28      1740
         1.0       0.56      0.81      0.67      2227

    accuracy                           0.54      3967
   macro avg       0.51      0.51      0.47      3967
weighted avg       0.52      0.54      0.50      3967
```


![modified svm](/Starter_Code/modified_svm_model_cumulative_return_plot.png)

### Training period modified to 6 months instead of 3 months

```python 
    precision    recall  f1-score   support

        -1.0       0.45      0.28      0.35      1651
         1.0       0.57      0.73      0.64      2120

    accuracy                           0.54      3771
   macro avg       0.51      0.51      0.49      3771
weighted avg       0.52      0.54      0.51      3771

```

![Adjusted svm](/Starter_Code/adjusted%20size%20smv%20model%20cumulative%20return%20plot.png)

## Evaluate a New Machine Learning Classifier

### Applying Logistic Regression instead of Support Vector Machines

Classification Report

```python
              precision    recall  f1-score   support

        -1.0       0.44      0.33      0.38      1804
         1.0       0.56      0.66      0.61      2288

    accuracy                           0.52      4092
   macro avg       0.50      0.50      0.49      4092
weighted avg       0.51      0.52      0.51      4092
```

![lr](/Starter_Code/LogisticRegression%20model%20cumulative%20return%20plot.png)

### Applying Decision tree classifier instead of Support Vector Machines

Classification Report -

```python
            precision    recall  f1-score   support

        -1.0       0.44      0.92      0.60      1804
         1.0       0.56      0.08      0.14      2288

    accuracy                           0.45      4092
   macro avg       0.50      0.50      0.37      4092
weighted avg       0.51      0.45      0.34      4092
```

![decision tree](/Starter_Code/Decision%20tree%20model-%20cumulative%20return.png)

### Applying AdaBoost instead of Support Vector Machines

Classification Report

```python 
           precision    recall  f1-score   support

        -1.0       0.44      0.08      0.13      1804
         1.0       0.56      0.92      0.70      2288

    accuracy                           0.55      4092
   macro avg       0.50      0.50      0.41      4092
weighted avg       0.51      0.55      0.45      4092
```

![adaboost](/Starter_Code/adaboost_cumulative_return_plot.png)

## Conclusion 

The baseline model worked well on finding the signals for long strategy that is to buy shares. Where as it didn’t perform great in finding signals for short sell. The Accuracy just went above 50%. Also, it might go in overfitting if it is used on unseen data. So the model need further tunning or considering a different model to predict short sell signal as well.

After trying on tunning the model by adjusting the SMA Windows, helped a little to get more signals for short sell but signals for buy stock went down to 81%. The model improved for short sell signals as compared to baseline model.

Next, try was to increase the training period to 6 months, showed some improvement in short sell signal but the buy signal suffered furthermore. Precision and accuracy stated same. 

Finally, applying different models results into same precision for all three models. Logistic Regression model was better at finding sell signals. Decision tree was opposite of baseline model that is great at finding sell short signals. AdaBoost was similar to baseline model. So overall model still needs to be optimized to perform well. 

## Contributor

Brought to you by Amrita Prithiani

## License

MIT
