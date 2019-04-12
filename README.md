# ML Helper
---
Helpers to speed up and structure machine learning projects.

The library is available in [Pypi](https://pypi.org/project/ml-helper/)

### Installing
---


The easiest way to install ml-helper is through ```pip```

```python
pip install ml-helper
```

To use it in your project, you must first import the library

```python
from ml_helper.helper import Helper
```

And then create a Helper object with a dictionary of keys related to your project

```python
KEYS = {
    'SEED': 1,
    'TARGET': 'y',
    'METRIC': 'r2',
    'TIMESERIES': True,
    'SPLITS': 5
}

hp = Helper(KEYS)
```

After this, you may use the helper object's many functions

#### Dependencies

ML-Helper requires:
* Python (>3.5)
* Numpy (>=1.16)
* Pandas (>=0.23.4)
* Seaborn (>=0.9)
* Scikit-learn (>=0.20)
* Natplotlib (>=3)
* Scipy (>=1)
* Imblearn
* Vecstack

### Functionality
---

The functionality is separated into 4 groups:
* Data Exploration
    * Missing Data
    * Boxplot of numerical variables
    * Coefficient of variation
    * Correlation (numerical and categorical)
    * Under Represented Features
    * Target Variable Distribution
    * Feature Importance
    * PCA Component Variance
* Data Preparation
    * Convert features to categories
    * Drop multiple columns
* Modeling
    * Cross Validation (with stratified kfolds, or time series split depending on use case)
        * Randomized Grid Search
    * Pipeline: Collection of models and pipeline steps that get performed and scored
    * Predict: Predict on unseen data
    * Stack Predict: Build a stacked model and perform a prediction
* Regression
    * Plots for predictions
* Classification
    * ROC Curve
    * Classification Report
* Others
    * Select features based on types
    * Split X and y
    * Plot models/pipelines

### Working Examples
---
If you wish to see the library in use, you may view the notebooks in the [examples](examples) section.

Also, you can see the implementation in their corresponding Kaggle Kernels:

* [Bike Sharing in Washington D.C.: Time Series Regression](https://www.kaggle.com/akoury/bike-sharing-in-washington-d-c-using-ml-helper)

* [Employee Attrition: Classification](https://www.kaggle.com/akoury/employee-attrition-basis-to-create-ml-helper-lib)

### ML-Helper Coding Style
---
Ml-Helper complies to PEP8 and uses ```black``` for coding standards

### Versioning
---
[SemVer](http://semver.org/) is used for versioning. 

### License
---
This project is licensed under the MIT License - see the [License](license.txt) file for details