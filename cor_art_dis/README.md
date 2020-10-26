heart_disease
==============================

Coronory Artery Disease Project
Exploratory analysis
Prediction of heart disease given 14 variables
For this analysis we use the Cleveland "Coronary Artery Disease" dataset found on the UCI Machine Learning Repository at the following location:

Heart Disease Dataset

The objective of the analysis is to use statistical learning to identify factors associated with Coronary Artery Disease as indicated by a coronary angiography interpreted by a Cardiologist.

According to the paper by (Detrano et al., 1989) the data represents data collected for 303 patients referred for coronary angiography at the Cleveland Clinic between May 1981 and September 1984. The 13 independent/ features variables can be divided into 3 groups as follows:

Routine evaluation (based on historical data):

ECG at rest
Serum Cholesterol
Fasting blood sugar
Non-invasive test data (informed consent obtained for data as part of research protocol):

Exercise ECG
ST-segment peak slope (upsloping, flat or downsloping)
ST-segment depression
Excercise Thallium scintigraphy (fixed, reversible or none)
Cardiac fluoroscopy (number of vessels appeared to contain calcium)
Other demographic and clinical variables (based on routine data):

Age
Sex
Chest pain type
Systolic blood pressure
ST-T-wave abnormality (T-wave abnormality)
Probably or definite ventricular hypertrophy (Este's criteria)
The dependent/ response variable was the angiographic test result indicating a >50% diameter narrowing.


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
