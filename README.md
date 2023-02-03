# FairAssign
This repository contains implementation of the FairAssign algorithm defined in our paper titled "Stochastically Fair Driver Assignment in Last Mile Delivery: From E-comm to Food Platforms", 

## Getting Started
These instructions will get you a copy of the project up and running on your local mahcine.

### Pre-requisites
The algorithms are implemented in Python 3 (Python 3.9)     
For solving the linear programs, the following LP Solvers will be needed: [Gurobi Optimizer](https://www.gurobi.com/downloads/) and [IBM Cplex](https://www.ibm.com/products/ilog-cplex-optimization-studio)

---

### E-commerce 
**Data** : Please download the Brazilian e-commerce dataset using any of these links [link1](https://csciitd-my.sharepoint.com/personal/csy217543_iitd_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fcsy217543%5Fiitd%5Fac%5Fin%2FDocuments%2Fecomm%5Fdata), [link2](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). Place the downloaded directory named 'olist' in [./e-commerce](e-commerce).     
**Code** : Please follow the notebook 'ecomm.ipynb' present in [./e-commerce](e-commerce) to reproduce the results for the e-commerce setting. 

---

### Food Delivery 
**Data** : To get the data, please follow the instructions given [here](onedrive link?).   
**Code** : Please follow the notebook 'food_dlvry.ipynb' present in [./food-delivery](food-delivery) to reproduce the results for the food delivery setting.

---
