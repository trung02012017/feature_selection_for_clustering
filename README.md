## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)

## General info
This project conducts an experiment on model selection for kmeans and agglomerative hierarchical clustering algorithms 
with a given dataset.
	
## Technologies
Project is created with:
* Python 3.9
	
## Setup
* Download data in the folder of case study 1 and extract it to ``data`` folder
* Run ``pip install -r requirements.txt`` to install required libraries
* Run ``python main.py --mode visualize`` to visualize data. The results of this process will be saved in 
``results/data_visualiztion`` folder 
* Run ``python main.py --mode train --model kmeans`` to train kmeans model on given dataset. The resutls of this 
process will be saved in ``results/kmeans`` folder
* Run ``python main.py --mode train --model ahc`` to train kmeans model on given dataset. The resutls of this 
process will be saved in ``results/ahc`` folder