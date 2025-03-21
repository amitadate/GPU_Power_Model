

```bash
cd /path/to/scaffold_directory

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

#  Histogram Power Model Training

## Overview
This project is focused on training a power model using CUDA and Python. The model aims to analyze power consumption data and produce histograms for better visualization of results. This README provides a guide to set up the necessary environment and execute the training process.

## Prerequisites
- Python 3.x
- CUDA (ensure compatible version is installed)
- Git (to clone repository if needed)

## Steps

clone the repository to your local machine. Replace the URL with the actual repository URL.

```bash
git clone https://github.com/yourusername/cuda-lab3.git
cd cuda-lab3

cd /path/to/scaffold_directory

python -m venv venv
source venv/bin/activate


pip install -r requirements.txt


./train_pm_full.sh


## All outputs will be saved in a folder named histogram_power_model_[day]_[time], where [day] and [time] represent the current date and time of execution
