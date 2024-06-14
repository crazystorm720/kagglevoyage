### KaggleVoyage

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

### Conda Environment Setup

Creating a `conda` environment using an `environment.yml` file is a great way to manage your project dependencies. Here's how you can set up a base `environment.yml` for your Kaggle projects:

#### Base `environment.yml`

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

#### Explanation of the Components

- **name**: The name of the conda environment.
- **channels**: The channels from which conda should fetch the packages. `conda-forge` is a popular community-driven channel.
- **dependencies**: The packages to be installed in the environment.
  - **python=3.9**: Specifies the Python version.
  - **pandas, numpy, matplotlib, seaborn, scikit-learn**: Core data science libraries.
  - **jupyterlab, ipykernel**: Tools for running Jupyter notebooks.
  - **xgboost, lightgbm, catboost**: Popular machine learning libraries.
  - **pip**: Ensures that pip is installed so additional packages can be added.
  - **kaggle**: Installed via pip to interact with the Kaggle API.

#### Creating the Environment

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

#### Using the Environment in Jupyter

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

#### Keeping Your Environment Up-to-Date

As you progress and need additional packages, you can update your `environment.yml` and re-create the environment or install the packages directly into the environment using `conda install` or `pip install`.

#### Example of Updating `environment.yml`

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

---

### Practicing CLI Skills

Practicing your CLI skills while working on the Kaggle competition is a great idea. Here are some key areas and commands to focus on:

#### 1. **Navigating the File System**

- **List Files and Directories**:
  ```sh
  ls -la
  ```
- **Change Directory**:
  ```sh
  cd /path/to/directory
  ```
- **Create Directories**:
  ```sh
  mkdir -p competitions/titanic/data/raw
  mkdir -p competitions/titanic/data/processed
  mkdir -p competitions/titanic/notebooks
  mkdir -p competitions/titanic/scripts
  mkdir -p competitions/titanic/models
  mkdir -p competitions/titanic/reports/figures
  ```
- **Move or Copy Files**:
  ```sh
  mv source destination
  cp source destination
  ```

#### 2. **Working with Git and GitHub**

- **Clone Repository**:
  ```sh
  git clone https://github.com/yourusername/kagglevoyage.git
  cd kagglevoyage
  ```
- **Initialize a Git Repository**:
  ```sh
  git init
  ```
- **Add Files to Staging Area**:
  ```sh
  git add .
  ```
- **Commit Changes**:
  ```sh
  git commit -m "Initial commit with project structure"
  ```
- **Push Changes to GitHub**:
  ```sh
  git remote add origin https://github.com/yourusername/kagglevoyage.git
  git push -u origin main
  ```
- **Check Git Status**:
  ```sh
  git status
  ```
- **View Git Log**:
  ```sh
  git log
  ```

#### 3. **Managing Conda Environments**

- **Create Environment from `environment.yml`**:
  ```sh
  conda env create -f environment.yml
  ```
- **Activate Environment**:
  ```sh
  conda activate kagglevoyage
  ```
- **List Environments**:
  ```sh
  conda env list
  ```
- **Update Environment**:
  ```sh
  conda env update -f environment.yml --prune
  ```

#### 4. **Using Kaggle CLI**

- **Install Kaggle CLI**:
  ```sh
  pip install kaggle
  ```
- **Set Up Kaggle API Credentials**:
  - Download your API token from Kaggle (Account settings).
  - Place the `kaggle.json` file in `~/.kaggle/`.

- **Download Competition Data**:
  ```sh
  kaggle competitions download -c titanic -p competitions/titanic/data/raw
  ```
- **List Competitions**:
  ```sh
  kaggle competitions list
  ```

#### 5. **Data Processing with CLI Tools**

- **View First Few Lines of a File**:
  ```sh
  head competitions/titanic/data/raw/train.csv
  ```
- **View Last Few Lines of a File**:
  ```sh
  tail competitions/titanic/data/raw/train.csv
  ```
- **Count Lines in a File**:
  ```sh
  wc -l competitions/titanic/data/raw/train.csv
  ```

#### 6. **Running Python Scripts**

- **Run a Python Script**:
  ```sh
  python scripts/data_cleaning.py
  ```
- **Jupyter Notebook via CLI**:
  ```sh
  jupyter notebook
  jupyter lab
  ```

#### Example Workflow

1. **Clone the Repository**:
   ```sh
   git clone https://github.com/yourusername/kagglevoyage.git
   cd kagglevoyage
   ```

2. **Create Directory Structure**:
   ```sh
   mkdir -p competitions/titanic/{data/raw,data/processed,notebooks,scripts,models,reports/figures}
   ```

3. **Create and Activate Conda Environment**:
   ```sh
   conda env create -f environment.yml
   conda activate kagglevoyage
   ```

4. **Download Kaggle Data**:
   ```sh
   kaggle competitions download -c titanic -p competitions/titanic/data/raw
   unzip competitions/titanic/data/raw/titanic.zip -d competitions/titanic/data/raw
   ```

5. **Run Data Processing Script**:
   ```sh
   python scripts/data_cleaning.py
   ```

6. **Commit and Push Changes**:
   ```sh
   git add .
   git commit -m "Added data cleaning script"
   git push origin main
   ```

#### Further Resources

- **Man Pages**: Use `man` command to learn about other commands:
  ```sh
  man ls
  man git
  man conda
  ```
- **Bash Scripting**: Automate repetitive tasks using Bash scripts.
- **Vim/Nano**: Learn to use CLI text editors for quick edits.

By practicing these commands and incorporating them into your workflow, you'll become proficient with the CLI and enhance your productivity in managing Kaggle competitions and other projects. If you need further guidance or have specific questions, feel free to ask!

---

### Advanced CLI Techniques and Tools

For more advanced CLI usage within the context of the Kaggle competition, you can leverage powerful command-line tools and scripting to enhance your workflow. Here are some advanced CLI techniques and tools you can use:

#### 1. **Using `awk` and `sed` for Data Processing**

- **Extract Specific Columns**:
  ```sh
  awk -F',' '{print $1, $2, $3}' competitions/titanic/data/raw/train.csv > output.csv
  ```

- **Replace Missing Values**:
  ```sh
  sed -i 's/,,/,NA,/g' competitions/titanic/data/raw/train.csv
  ```

- **Filter Rows Based on a Condition**:
  ```sh
  awk -F',' '$3 > 30' competitions/titanic/data/raw/train.csv > filtered.csv
  ```

#### 2. **Using `jq` for JSON Processing**

- **Parse JSON Output**:
  ```sh
  cat data.json | jq '.data'
  ```

- **Filter JSON Data**:
  ```sh
  cat data.json | jq '.data[] | select(.age > 30)'
  ```

#### 3. **Using `parallel` for Parallel Execution**

- **Run Python Scripts in Parallel**:
  ```sh
  parallel python ::: script1.py script2.py script3.py
  ```

- **Process Files in Parallel**:
  ```sh
  find competitions/titanic/data/raw -name "*.csv" | parallel gzip
  ```

#### 4. **Using `tmux` for Session Management**

- **Start a New tmux Session**:
  ```sh
  tmux new -s kaggle
  ```

- **Detach and Reattach to a Session**:
  ```sh
  tmux detach
  tmux attach -t kaggle
  ```

- **Split tmux Window**:
  ```sh
  tmux split-window -h
  ```

#### 5. **Advanced `git` Usage**

- **Interactive Rebase**:
  ```sh
  git rebase -i HEAD~3
  ```

- **Stash Changes**:
  ```sh
  git stash save "WIP"
  git stash apply
  ```

- **Squash Commits**:
  ```sh
  git rebase -i HEAD~3
  # In the editor, change 'pick' to 'squash' for the commits you want to squash
  ```

#### 6. **Using `Make` for Build Automation**

- **Create a Makefile**:
  ```Makefile
  all: data_clean process_data train_model

  data_clean:
    python scripts/data_cleaning.py

  process_data:
    python scripts/process_data.py

  train_model:
    python scripts/train_model.py
  ```

- **Run Makefile**:
  ```sh
  make
  ```

#### 7. **Using `curl` and `wget` for Data Download**

- **Download Data with `curl`**:
  ```sh
  curl -o train.csv https://path/to/train.csv
  ```

- **Download Data with `wget`**:
  ```sh
  wget https://path/to/train.csv
  ```

#### 8. **Using `rsync` for Efficient File Sync**

- **Sync Directories**:
  ```sh
  rsync -avz --progress source/ destination/
  ```

#### 9. **Advanced Scripting with Bash**

- **Automate Data Processing**:
  ```bash
  #!/bin/bash
  # data_processing.sh

  # Download Data
  kaggle competitions download -c titanic -p competitions/titanic/data/raw

  # Unzip Data
  unzip competitions/titanic/data/raw/titanic.zip -d competitions/titanic/data/raw

  # Run Data Cleaning Script
  python scripts/data_cleaning.py

  # Run Data Processing Script
  python scripts/process_data.py

  # Train Model
  python scripts/train_model.py

  echo "Data processing and model training completed."
  ```

- **Run the Script**:
  ```sh
  chmod +x data_processing.sh
  ./data_processing.sh
  ```

#### Example Workflow with Advanced CLI Techniques

1. **Clone the Repository and Navigate to the Project Directory**:
   ```sh
   git clone https://github.com/yourusername/kagglevoyage.git
   cd kagglevoyage
   ```

2. **Create Directory Structure**:
   ```sh
   mkdir -p competitions/titanic/{data/raw,data/processed,notebooks,scripts,models,reports/figures}
   ```

3. **Create and Activate Conda Environment**:
   ```sh
   conda env create -f environment.yml
   conda activate kagglevoyage
   ```

4. **Download Kaggle Data**:
   ```sh
   kaggle competitions download -c titanic -p competitions/titanic/data/raw
   unzip competitions/titanic/data/raw/titanic.zip -d competitions/titanic/data/raw
   ```

5. **Use `awk` to Inspect Data**:
   ```sh
   awk -F',' '{print $1, $2, $3}' competitions/titanic/data/raw/train.csv | head
   ```

6. **Use `sed` to Clean Data**:
   ```sh
   sed -i 's/,,/,NA,/g' competitions/titanic/data/raw/train.csv
   ```

7. **Run Python Scripts in Parallel**:
   ```sh
   parallel python ::: scripts/data_cleaning.py scripts/process_data.py scripts/train_model.py
   ```

8. **Use `tmux` to Manage Sessions**:
   ```sh
   tmux new -s kaggle
   ```

9. **Automate Workflow with Bash Script**:
   ```bash
   #!/bin/bash
   # data_processing.sh

   # Download Data
   kaggle competitions download -c titanic -p competitions/titanic/data/raw

   # Unzip Data
   unzip competitions/titanic/data/raw/titanic.zip -d competitions/titanic/data/raw

   # Run Data Cleaning Script
   python scripts/data_cleaning.py

   # Run Data Processing Script
   python scripts/process_data.py

   # Train Model
   python scripts/train_model.py

   echo "Data processing and model training completed."
   ```

   - **Run the Script**:
     ```sh
     chmod +x data_processing.sh
     ./data_processing.sh
     ```

---

### Remote ML Workstation Setup

To focus on using a remote ML workstation and ensuring that everything works headlessly, follow these steps:

#### 1. **Set Up the Remote Server**

- **Access the Remote Server**:
  ```sh
  ssh user@remote_server_ip
  ```

- **Install Conda**:
  ```sh
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh
  ```

- **Clone the Repository**:
  ```sh
  git clone https://github.com/yourusername/kagglevoyage.git
  cd kagglevoyage
  ```

- **Create and Activate Conda Environment**:
  ```sh
  conda env create -f environment.yml
  conda activate kagglevoyage
  ```

#### 2. **Run Jupyter Notebook/Lab Headlessly**

- **Install Jupyter Notebook Extensions**:
  ```sh
  conda install -c conda-forge jupyter_contrib_nbextensions
  jupyter contrib nbextension install --user
  ```

- **Start

 Jupyter Notebook or Lab**:
  ```sh
  jupyter lab --no-browser --port=8888
  ```
  or
  ```sh
  jupyter notebook --no-browser --port=8888
  ```

- **Set Up SSH Tunneling**:
  On your local machine, run:
  ```sh
  ssh -N -f -L localhost:8888:localhost:8888 user@remote_server_ip
  ```

- **Access Jupyter in Browser**:
  Open your browser and go to `http://localhost:8888`.

#### 3. **Run Scripts Remotely**

- **Run Data Processing Script**:
  ```sh
  ssh user@remote_server_ip "cd /path/to/kagglevoyage && conda activate kagglevoyage && python scripts/data_cleaning.py"
  ```

- **Use `tmux` to Manage Long-Running Processes**:
  ```sh
  ssh user@remote_server_ip
  tmux new -s kaggle
  cd /path/to/kagglevoyage
  conda activate kagglevoyage
  python scripts/train_model.py
  [Ctrl+B, D] # Detach the tmux session
  exit
  ```

- **Reattach to the tmux Session**:
  ```sh
  ssh user@remote_server_ip
  tmux attach -t kaggle
  ```

#### 4. **Automate Workflow with Bash Script**

- **Create a Bash Script**:
  ```bash
  #!/bin/bash
  # remote_workflow.sh

  # Clone the repository
  git clone https://github.com/yourusername/kagglevoyage.git
  cd kagglevoyage

  # Create and activate conda environment
  conda env create -f environment.yml
  conda activate kagglevoyage

  # Download Kaggle data
  kaggle competitions download -c titanic -p competitions/titanic/data/raw
  unzip competitions/titanic/data/raw/titanic.zip -d competitions/titanic/data/raw

  # Run data processing script
  python scripts/data_cleaning.py

  # Run data processing script
  python scripts/process_data.py

  # Train model
  python scripts/train_model.py

  echo "Data processing and model training completed."
  ```

- **Run the Bash Script on the Remote Server**:
  ```sh
  ssh user@remote_server_ip "bash /path/to/remote_workflow.sh"
  ```

By following these steps, you can effectively manage your Kaggle projects using a remote ML workstation, ensuring everything works headlessly and efficiently. If you need further customization or have additional requirements, feel free to ask!

---

```python
import os

def create_directory_structure(base_path):
    competitions = ['titanic', 'competition_2']  # Add more competition names as needed

    for competition in competitions:
        paths = [
            f"{base_path}/competitions/{competition}/data/raw",
            f"{base_path}/competitions/{competition}/data/processed",
            f"{base_path}/competitions/{competition}/notebooks",
            f"{base_path}/competitions/{competition}/scripts",
            f"{base_path}/competitions/{competition}/models",
            f"{base_path}/competitions/{competition}/reports/figures"
        ]

        for path in paths:
            os.makedirs(path, exist_ok=True)

    shared_paths = [
        f"{base_path}/shared/scripts",
        f"{base_path}/shared/notebooks",
        f"{base_path}/shared/data"
    ]

    for path in shared_paths:
        os.makedirs(path, exist_ok=True)

    print(f"Directory structure created under {base_path}")

base_path = 'kagglevoyage'
create_directory_structure(base_path)
```