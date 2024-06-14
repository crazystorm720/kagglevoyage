
# KaggleVoyage

Documenting my data science journey through Kaggle competitions. Follow along as I explore datasets, build models, and share insights and learnings.

## Project Structure

```
kagglevoyage/
├── competitions/
│   ├── titanic/
│   │   ├── data/
│   │   │   ├── raw/
│   │   │   └── processed/
│   │   ├── notebooks/
│   │   ├── scripts/
│   │   ├── models/
│   │   └── reports/
│   │       └── figures/
│   ├── competition_2/
│   │   ├── data/
│   │   │   ├── raw/
│   │   │   └── processed/
│   │   ├── notebooks/
│   │   ├── scripts/
│   │   ├── models/
│   │   └── reports/
│   │       └── figures/
│   └── .../
├── shared/
│   ├── scripts/
│   ├── notebooks/
│   └── data/
├── README.md
├── requirements.txt
└── .gitignore
```

## How to Use

1. **Clone the Repository**
   ```sh
   git clone https://github.com/yourusername/kagglevoyage.git
   cd kagglevoyage
   ```

2. **Install Dependencies**
   Install the necessary Python libraries:
   ```sh
   pip install -r requirements.txt
   ```

3. **Download Data**
   Place the raw data files in the appropriate `data/raw/` directory.

4. **Run Notebooks and Scripts**
   Follow along with the Jupyter notebooks in the `notebooks/` directory to see the data exploration and modeling steps.

## Competitions

- [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)
- More to come...

## Contributing

Feel free to submit issues, fork the repository, and make pull requests. Any contributions are welcome!

## License

This project is licensed under the MIT License.

---

Creating a `conda` environment using an `environment.yml` file is a great way to manage your project dependencies. Here's how you can set up a base `environment.yml` for your Kaggle projects:

### Base `environment.yml`

This file will include the essential libraries for data science and machine learning, as well as some commonly used tools.

```yaml
name: kagglevoyage
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - jupyterlab
  - ipykernel
  - xgboost
  - lightgbm
  - catboost
  - pip
  - pip:
      - kaggle
```

### Explanation of the Components

- **name**: The name of the conda environment.
- **channels**: The channels from which conda should fetch the packages. `conda-forge` is a popular community-driven channel.
- **dependencies**: The packages to be installed in the environment.
  - **python=3.9**: Specifies the Python version.
  - **pandas, numpy, matplotlib, seaborn, scikit-learn**: Core data science libraries.
  - **jupyterlab, ipykernel**: Tools for running Jupyter notebooks.
  - **xgboost, lightgbm, catboost**: Popular machine learning libraries.
  - **pip**: Ensures that pip is installed so additional packages can be added.
  - **kaggle**: Installed via pip to interact with the Kaggle API.

### Creating the Environment

1. **Save the `environment.yml` File**
   Save the above content into a file named `environment.yml`.

2. **Create the Environment**
   Use the following command to create the environment from the `environment.yml` file:
   ```sh
   conda env create -f environment.yml
   ```

3. **Activate the Environment**
   After creating the environment, activate it using:
   ```sh
   conda activate kagglevoyage
   ```

4. **Verify the Installation**
   Ensure all packages are installed correctly:
   ```sh
   conda list
   ```

### Using the Environment in Jupyter

To use the new conda environment in Jupyter notebooks, you need to set up an IPython kernel:

1. **Install the IPython Kernel in the Environment**
   ```sh
   conda activate kagglevoyage
   python -m ipykernel install --user --name kagglevoyage --display-name "Python (kagglevoyage)"
   ```

2. **Start JupyterLab**
   ```sh
   jupyter lab
   ```

3. **Select the Kernel**
   When you open a new notebook in JupyterLab, select the "Python (kagglevoyage)" kernel.

### Keeping Your Environment Up-to-Date

As you progress and need additional packages, you can update your `environment.yml` and re-create the environment or install the packages directly into the environment using `conda install` or `pip install`.

### Example of Updating `environment.yml`

If you need to add a new package, say `tensorflow`, you can update your `environment.yml` like this:

```yaml
name: kagglevoyage
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - jupyterlab
  - ipykernel
  - xgboost
  - lightgbm
  - catboost
  - tensorflow
  - pip
  - pip:
      - kaggle
```

Then update the environment with:

```sh
conda env update -f environment.yml --prune
```

This will add the new package and remove any dependencies that are no longer required.

By following this setup, you'll have a robust and flexible environment for tackling Kaggle competitions. If you need further customization or have additional requirements, feel free to ask!