# Predictive Maintenance Tutorial

![Data_Diagram](https://www.usb-antivirus.com/wp-content/uploads/2014/11/tutorial-windwos-10-2-320x202.png)

* Documentation site for Microsoft internal dogfooders.
* Documentation site for external private preview customers.

Leave the image icon and the document links as what it is right now. We will update later.

**The above info will be included in the Readme on GitHub**

## Prerequisites

Please note this section will be included in the Readme file on GitHub repo.

1. Ensure that you have properly installed Azure ML Workbench by following the [installation guide](https://github.com/Azure/ViennaDocs/blob/master/Documentation/Installation.md).
2. For [operationalization](https://github.com/Azure/Machine-Learning-Operationalization) of the model, it is best if the user runs a [Docker engine](https://www.docker.com/) installed and running locally. If not, you can use the cluster option but be aware that running an [Azure Container Service (ACS)](https://azure.microsoft.com/en-us/services/container-service/) can often be expensive.
3. Intermediate results for use across Jupyter notebooks in this tutorial is stored in an Azure Blob Storage container. Instructions for setting up an Azure Storage account are available at this [link](https://docs.microsoft.com/en-us/azure/storage/blobs/storage-python-how-to-use-blob-storage). 
4. This tutorial assumes that the user is running Azure ML Workbench on a Windows 10 machine with Docker engine locally installed. 

Note: The tutorial was built and tested on a Windows 10 machine with the following specification: Intel Core i7-4600U CPU @ 2.10GHz, 8GB RAM, 64-bit OS, x64-based processor with Docker Version 17.06.0-ce-win19 (12801).

## Tutorial Introduction

The impact of unscheduled equipment downtime can be extremely detrimental for any business. It is critical to therefore keep field equipment running in order to maximize utilization and performance and by minimizing costly, unscheduled downtime. Early identification of issues can help allocate limited maintenance resources in a cost-effective way and enhance quality and supply chain processes. 

For this tutorial we will use [large scale data](https://github.com/Microsoft/SQL-Server-R-Services-Samples/tree/master/PredictiveMaintanenceModelingGuide/Data) and then walk the user through the main steps from data ingestion, feature engineering, model building and then finally model operationalization and deployment. The code for the entire process is written in PySpark and implemented using Jupyter notebooks in Azure ML Workbench. The best model is finally operationalized using using Azure Machine Learning Model Management environment for use in production for making realtime failure predictions.   


## Use Case Overview

A major problem faced by businesses in asset-heavy industries is the significant costs that are associated with delays to mechanical problems. Most businesses are interested in predicting when these problems will arise in order to proactively prevent them before they occur. This will reduce the costs by reducing downtime and, in some cases, increasing safety. Please refer to the [playbook for predictive maintenance](https://docs.microsoft.com/en-us/azure/machine-learning/cortana-analytics-playbook-predictive-maintenance) for a detailed explanation of common use cases and the modeling approach for predictive maintenance.

This tutorial leverages the ideas from the playbook with the aim of providing the steps to implement a predictive model for a scenario which is based on a synthesis of multiple real-world business problems. This example brings together common data elements observed among many predictive maintenance use cases.

The business problem for this simulated data is to predict issues caused by component failures. The business question therefore is “*What is the probability that a machine will go down due to failure of a component*?” This problem is formatted as a multiclass classification problem (multiple components per machine) and a machine learning algorithm is used to create the predictive model. The model is trained on historical data collected from machines. In this tutorial the user will go through the various steps of implementing such a model within the Azure Machine Learning Workbench environment.


## Data Description

The [simulated data](https://github.com/Microsoft/SQL-Server-R-Services-Samples/tree/master/PredictiveMaintanenceModelingGuide/Data) consists of 5 main CSV files. 

* [Machines](https://pdmmodelingguide.blob.core.windows.net/pdmdata/machines.csv): Features differentiating each machine. For example age and model.
* [Error](https://pdmmodelingguide.blob.core.windows.net/pdmdata/errors.csv): The error log contains are non-breaking errors thrown while the machine is still operational. These errors are not considered as failures, though they may be predictive of a future failure event. The error date-time are rounded to the closest hour since the telemetry data (loaded later) is collected on an hourly rate
* [Maintenance](https://pdmmodelingguide.blob.core.windows.net/pdmdata/maint.csv): The maintenance log contains both scheduled and unscheduled maintenance records. Scheduled maintenance corresponds with regular inspection of components, unscheduled maintenance may arise from mechanical failure or other performance degradation. 
* [Telemetry](https://pdmmodelingguide.blob.core.windows.net/pdmdata/telemetry.csv): The telemetry time-series data consists of voltage, rotation, pressure, and vibration sensor measurements collected from each machines in real time. The data is averaged over an hour and stored in the telemetry logs
* [Failures](https://pdmmodelingguide.blob.core.windows.net/pdmdata/failures.csv): Failures correspond to component replacements within the maintenance log. Each record contains the Machine ID, component type, and replacement date and time. These records will be used to create the machine learning labels we will be trying to predict.

See the [Data Ingestion](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_git/ViennaPredMaintTut?path=%2FCode%2Fdata_ingestion.ipynb&version=GBmaster&_a=contents) Jupyter Notebook tutorial in Code section to download the raw data sets from the GitHub repository and create the PySpark data sets for this analysis.

## Tutorial Structure
The content for the tutorial is available at the [GitHub repository](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_git/ViennaPredMaintTut?_a=contents&path=%2F&version=GBmaster). 

In the repository there is a [Readme](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_git/ViennaPredMaintTut?_a=preview&path=%2FREADME.md&version=GBmaster) file outlines the processes from preparing the data till building a few model and then finally operationalization of the best model. All the Jupyter notebooks are available in the [Code](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_git/ViennaPredMaintTut?path=%2FCode&version=GBmaster&_a=contents) folder.   

Next we describe the step-by-step tutorial workflow. The end to end tutorial is written in PySpark and is split into 4 notebooks as outlined below:

* [Data Ingestion](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_git/ViennaPredMaintTut?path=%2FCode%2Fdata_ingestion.ipynb&version=GBmaster&_a=contents): this notebook handles the data ingestion of the 5 input files, does some preliminary cleanup, creates some summary graphics to verify the data download, and finally stores the resulting data sets in the Azure blob container for use within the next notebook.

* [Feature Engineering](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_git/ViennaPredMaintTut?path=%2FCode%2Ffeature_engineering.ipynb&version=GBmaster&_a=contents): using the cleaned dataset from the previous step, lag features are created for the telemetry sensors, and additional feature engineering is done to create variables like days since last replacement etc. Finally the failures are tagged to the relevant rows of data to create a final dataset which is saved in Azure blob for the next step. 

* [Model Building](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_git/ViennaPredMaintTut?path=%2FCode%2Fmodel_building.ipynb&version=GBmaster&_a=contents): the final feature engineered dataset is then split into two namely train/test datasets based on a date-time stamp. Then two models namely a Random Forest Classifier and Decision Tree Classifier are built on the training dataset and then scored on the test dataset. 

* [Model operationalization & Deployment](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_git/ViennaPredMaintTut?path=%2FCode%2Ftest_pdm_operationalization.ipynb&version=GBmaster&_a=contents): the best model built in the previous step can be saved as a .model file along with the relevant scheme for deployment. The functions are first tested locally before operationalizing the model using Azure Machine Learning Model Management environment for use in production in realtime.   

## Conclusion & Next Steps

This tutorial gives the reader an overview of how to build an end to end predictive maintenance use case using PySpark within the Jupyter notebook environment in Azure ML Workbench. Then the final best model can be easily operationalized and deployed using Azure Machine Learning Model Management environment for use in production in realtime/.  

## References

This use case has been previously developed on multiple platforms as listed below:

* [Predictive Maintenance Solution Template](https://docs.microsoft.com/en-us/azure/machine-learning/cortana-analytics-playbook-predictive-maintenance)
* [Predictive Maintenance Modelling Guide](https://gallery.cortanaintelligence.com/Collection/Predictive-Maintenance-Modelling-Guide-1)
* [Predictive Maintenance Modeling Guide using SQL R Services](https://gallery.cortanaintelligence.com/Tutorial/Predictive-Maintenance-Modeling-Guide-using-SQL-R-Services-1)
* [Predictive Maintenance Modelling Guide Python Notebook](https://gallery.cortanaintelligence.com/Notebook/Predictive-Maintenance-Modelling-Guide-Python-Notebook-1)
* [Predictive Maintenance using PySpark](https://gallery.cortanaintelligence.com/Tutorial/Predictive-Maintenance-using-PySpark)

## Acknowledgement

Multiple team members from across various organizations within Microsoft worked collaboratively to help build this end to end tutorial.  

## Contact

###Please feel free to contact Your Name (yourname@microsoft.com) with any question or comment.

## Disclaimer

###Leave this session as what it is for now. We will update the content once we get more concrete answers from the legal team.