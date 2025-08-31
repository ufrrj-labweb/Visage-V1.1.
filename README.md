Visage
==============================

Code and images used for the VISAGE research project.

V1.1 version, made to adress some issues in the original code

Model available on hugging faces:
<https://huggingface.co/MHCTDS/visage>

The Python version is 3.8, for extra stability at the cost of speed.

If the data is not available, we might not have released it yet for public use, as it contains sensitive data in the form of user ids.

If your aim is not to reproduce the results but just use the code to learn, the content of this repository with no data is enough.

Acellerate is a library that allows for the automatic parallelization of code using multiple CPUs, GPUs and NPUs based on the configuration you use, with the option of configuring your own settings or letting it auto recognize your hardware on installation via pip.

Requirements was made for mac, so it can't be installed directly to other systems.

Tested on python 3.12 on linux for cuda, but needs a few adjustments to the functions to natively run on a cuda gpu.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
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
    │   └── figures        <- Generated graphics and figures to be used in reporting  // Contains the images used on the paper
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`                            
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── DataProcessing.py  // Class for data processing
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── metrics       <- Scripts to evaluate scikit models and everything related to metrics
    │   │   └── MetricsProcessing.py  // Class for evaluation of models
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── ModelProcessing.py  // Factory for scikit model creation and partial validation
    │   │   ├── BertProcessing.py  // Independent class for all functions related for running the BERT model
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
