Visage
==============================

Code and images used for the VISAGE research project. Everything can be found on the notebooks exploratory folder, under the name boost_classifiers.
Model now available on hugging faces:
https://huggingface.co/MHCTDS/visage

The Python version is 3.11.9 
Please use:
Transformers=4.39.1
Accelerate=0.27.2

A dockerfile or docker-compose should be available and working, but if its not, take a look at the requirements alt for the version of the libraries you will use, keep in mind that not all will be the same as i am using a m2 max base mac studio on MacOS Sequoia 15.3.1. 
Downloading the Python, Transformers and Acellerate versions above, besides the libraries on the notebook with the versions on the requirements_alt should be enough to run the code anywhere though, exact instructions on P.S below.

If the data is not available, we might not have released it yet for public use, as it contains sensitive data in the form of user ids. If your aim is not to reproduce the results but just use the code to learn, the content of this repository with no data is enough. Although, I do not recommend using this code as a example of good optimization.

Some design choices might seem questionable at best with some of the functions here, but I had to make it so all the models would be judged fairly against each other and have enough versatility that this does not become legacy tech at our lab (it will though).

Acellerate is a library that allows for the automatic parallelization of code using multiple CPUs, GPUs and NPUs based on the configuration you use, with the option of configuring your own settings or letting it auto recognize your hardware on installation via pip.

P.S: if you delete the mac specific libraries on the requirements_alt and download it might work, but we didn't get the chance to test that hypothesis out. If it doesnt, run the libray imports on the notebook and download the error message library with the version on requirements_alt.

There might be some text artifacts left on the notebook from prior testing, since there was a time where we tested using or not normalizers for text, and we almost ran a XGBoost model, but it always crashed the macs kernel.


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
    │   └── figures        <- Generated graphics and figures to be used in reporting                    // Contains the images used on the paper
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`                            // requirements.txt is using the libraries installed on the docker, requirements_alt is the dependencies on mac
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
