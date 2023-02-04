# FairAssign
This repository contains implementation of the FairAssign algorithm defined in our paper titled "Stochastically Fair Driver Assignment in Gig Delivery Platforms", 

## Getting Started
These instructions will get you a copy of the project up and running on your local mahcine.

### Pre-requisites
The algorithms are implemented in Python 3 (Python 3.9)     
For solving the linear programs, the following LP Solvers will be needed: [Gurobi Optimizer](https://www.gurobi.com/downloads/) and [IBM Cplex](https://www.ibm.com/products/ilog-cplex-optimization-studio)

### Installation 
Setup a conda environment using the environment.yml file
```bash
conda env create -f environment.yml
conda activate fair_assign
```

---

### E-commerce 
**Data** : Please download the Brazilian e-commerce dataset using this [link](link-to-olist). Place the downloaded directory named 'olist' in [./e-commerce](e-commerce). On downloading, you will get a zip file named 'olist.zip'. Unzip 'olist.zip' to get the data directory named 'olist'. Please place the directory in ./ecommerce.
**Code** : Please follow the notebook 'ecomm.ipynb' present in [./e-commerce](e-commerce) to reproduce the results for the e-commerce setting. 

---

### Food Delivery 
**Data** : To get the data, please follow the instructions given [here](link_to_data_required_for_assignment; for_simulation_make_a_separate_request). Upon downloading you will get a zip file 'data.zip'. Please unzip it to obtain the data directory 'data' and place it in ./food-delivery.
**Code** : Please follow the notebook 'food_dlvry.ipynb' present in [./food-delivery](food-delivery) to reproduce the results for the food delivery setting.

---

