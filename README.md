# Support Vector Machines for HEP

This repository contains an implementation of support vector machines for high energy physics.

## Framework installation


To install, you need to log to a linux machine (see below) and in the terminal type:

1. Clone the repository:
  ```
  git clone git@github.com:andrex-naranjas/SVM-physics.git
  ```
2. Access the code:
  ```
  cd SVM-physics
  ```

3. Install the conda enviroment:
  ```
  conda env create -f config.yml
  conda activate vector
  conda develop .
  ```
3.1 Update the conda enviroment:
   ```
   conda env update --file config.yml --prune
   ```

4. Try your first example:
  ```
  python3 ./scripts/svm_example.py
  ```

5. Batch jobs using HTCondor is supported through the scripts
  ```
  batch_stats_summary.py
  submit_stats_svm.py
  ```
