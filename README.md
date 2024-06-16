# FairAssign
Codebase for our work titled ["FairAssign: Stochastically Fair Driver Assignment in Gig Delivery Platforms"](https://dl.acm.org/doi/10.1145/3593013.3594040) (FAccT 2023).

## Getting Started
These instructions will get you a copy of the project up and running on your local machine.

## Pre-requisites
The algorithms are implemented in Python 3 (Python 3.9)     
For solving the linear programs, the following LP Solvers will be needed: [Gurobi Optimizer](https://www.gurobi.com/downloads/) and [IBM Cplex](https://www.ibm.com/products/ilog-cplex-optimization-studio)

## Installation 
Setup a conda environment using the environment.yml file
```bash
conda env create -f environment.yml
conda activate fair_assign
```

---

## E-commerce 
**Data** : The data files relevant for assignment are present in ./e-commerce/data/. These files have been obtained [here](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). 

**Code** : Please follow the notebook 'ecomm.ipynb' present in [./e-commerce](e-commerce) to reproduce the results for the e-commerce setting. 

---

## Food Delivery 
**Data** : The food-delivery dataset is confidential. It is available on request.

**Code** : Please follow the notebook 'food_dlvry.ipynb' present in [./food-delivery](food-delivery) to reproduce the results for the food delivery setting.

---

## BibTex (Citation)
If you find our work useful, please cite using:
```
@inproceedings{10.1145/3593013.3594040,
  author       =  {Singh, Daman Deep and Das, Syamantak and Chakraborty, Abhijnan},
  title        =  {FairAssign: Stochastically Fair Driver Assignment in Gig Delivery Platforms},
  year         =  {2023},
  isbn         =  {9798400701924},
  publisher    =  {Association for Computing Machinery},
  address      =  {New York, NY, USA},
  url          =  {https://doi.org/10.1145/3593013.3594040},
  doi          =  {10.1145/3593013.3594040},
  booktitle    =  {Proceedings of the 2023 ACM Conference on Fairness, Accountability, and Transparency},
  pages        =  {753–763},
  numpages     =  {11},
  keywords     =  {Dependent Rounding, Ecommerce Logistics, Fair Driver Assignment, Food Delivery., Last Mile Delivery, Stochastic Fairness},
  location     =  {<conf-loc>, <city>Chicago</city>, <state>IL</state>, <country>USA</country>, </conf-loc>},
  series       =  {FAccT '23}
}
```
