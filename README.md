# Advanced Tutorial: General Predictive Maintenance
![](media/042116_1633_PredictiveM1.png "Predictive Maintenance")

* Documentation site for Microsoft internal dog food.
* Documentation site for external private preview customers.

## Introduction

Understanding fleet maintenance requirements can have a large impact on business safety and profitability. 

An initial approach is to rely on **corrective maintenance**, where parts are replaced as they fail. Corrective maintenance ensures parts are used completely (not wasting component life), but incurs expense in both downtime and unscheduled maintenance requirements (off hours, or inconvenient locations).

An alternative is a **preventative maintenance** schedule. Here a business may track or test component failures and determine a safe lifespan in which to replace that component before failure. For safety critical machinery, this approach can insure no catastrophic failures. The down side is components are replaced frequently, many with remaining life left. 

The goal of **predictive maintenance** is to optimize the balance between corrective and preventative maintenance. This approach only replaces those components when they are close to failure. The savings in this case come from both extending component lifespans (compared to preventive maintenance), and reducing unscheduled maintenance (over corrective maintenance) and improving safety associated component failure.

The goal of this scenario is to guide a data scientist through the implementation and operationalization of the predictive maintenance solution using *Azure Machine Learning Workbench*. 

## Prerequisites

  1. An installation of *Azure Machine Learning Workbench* Ap + CLI by following the [installation guide](../Installation.md).

  2. For operationalization, we require SSH access to a Linux VM with Docker engine installed. A convenient approach is to use an [Ubuntu Linux DSVM (Data Science Virtual Machine)](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/microsoft-ads.linux-data-science-vm-ubuntu). Instructions for using the CLI to  [Operationalization your model on a VM](http://aka.ms/o16ncli) are also provided.

  3. Intermediate data files are used to pass results between steps of our analysis. You will  _Create an Azure storage account_ for this storage. Detailed instructions for creating this account are at https://docs.microsoft.com/en-us/azure/storage/blobs/storage-python-how-to-use-blob-storage.

  
## Let's Begin

Now, launch the *Azure Machine Learning Workbench* App, sign-in, and create a new blank project. We can name it "PredictiveMaintenance."

## 0. Business Understanding

A major problem faced by businesses in asset-heavy industries is the significant costs that are associated with delays to mechanical problems. Most businesses are interested in predicting failures in order to proactively prevent them before they occur. Preventive maintenance reduces the costs by reducing downtime and, in some cases, increasing safety. Refer to the [playbook for predictive maintenance](https://docs.microsoft.com/en-us/azure/machine-learning/cortana-analytics-playbook-predictive-maintenance) for a detailed explanation of common use cases and the modeling approaches for predictive maintenance.

In this tutorial, we follow the ideas from the playbook and aim to provide the steps to implement a predictive model for a scenario based on a synthesis of multiple real-world business problems. This example brings together common data elements observed among many predictive maintenance use cases. 

The business problem for this simulated data is to predict issues caused by component failures. The business question “What is the probability that a machine goes down due to failure of a component?” This problem is formatted as a multi-class classification problem (multiple components per machine) and a machine learning algorithm is used to create the predictive model. The model is trained on historical data collected from machines. We go through the steps of implementing a model within the Azure Machine Learning Workbench. 

## Task 1. Prepare your data

The simulated data for this example comes from the following sources:

  * Machines: Features differentiating each machine. For example, age and model.
  * Error: The log of non-critical errors. These errors may still indicate an impending component failure.
  * Maint: Machine maintenance history detailing component replacement or regular maintenance activities withe the date of replacement.
  * Telemetry: The operating conditions of a machine from sensor data.
  * Failure history: The failure history of a machine or component within the machine.

We have stored the data on a GitHub site for other tutorials. To download this data and prepare your environment, start the Azure Machine Learning Workbench app, and open the Predictive Maintenance project. From a command line, run the download.py script with the command:
```
az ml experiment submit -c docker download.py
```
This script will take a few minutes to download and store the csv files in the shared folder (option 2 in https://github.com/Azure/ViennaDocs/blob/master/Documentation/PersistingChanges.md)

Once the data is downloaded, see the Data Ingestion Jupyter Notebook task in `Code/data_ingestion.ipnyb`, which loads the component data sets into PySpark format for this analysis. The raw data is stored in an Azure Blob storage container on your subscription for use in the feature engineering task.

## Task 2. Feature Engineering

See the Feature Engineering Jupyter Notebook task in `Code\feature_enginerring.ipnyb`, that takes PySpark data sets and creates the time series features used in the modeling step for this analysis. This feature engineering data set is also stored in your Azure Blob storage container for use in the model building and evaluation task.

## Task 3. Model Building & Evaluation

See the Model Building Jupyter Notebook task in `Code/model_building.ipnyb` that takes PySpark feature engineering data set from blob storage and builds and evaluates multiple models. It then compares these models to determine a "best" solution for predict component failures. The resulting model is serialized and stored in your Azure Blob storage container for use in the operationalization task.

## Task 4. Operationalization

## Conclusion

This real world scenario showcases an end to end predictive maintenance use case using the Jupyter notebook environment within Azure ML Workbench. We demonstrate using Jupyter notebooks to download data sources, engineer model features, and compare modeling techniques. We then work through how to deploy the model with the "best" performance, using Azure ML CLI.

## References

This use case has been previously developed on multiple platforms:
 
 * [Predictive Maintenance Solution Template](https://docs.microsoft.com/en-us/azure/machine-learning/cortana-analytics-playbook-predictive-maintenance)
 * [Predictive Maintenance Modeling Guide](https://gallery.cortanaintelligence.com/Collection/Predictive-Maintenance-Modelling-Guide-1)
 * [Predictive Maintenance Modeling Guide using SQL R Services](https://gallery.cortanaintelligence.com/Tutorial/Predictive-Maintenance-Modeling-Guide-using-SQL-R-Services-1)
 * [Predictive Maintenance Modeling Guide Python Notebook](https://gallery.cortanaintelligence.com/Notebook/Predictive-Maintenance-Modelling-Guide-Python-Notebook-1)
 * [Predictive Maintenance using PySpark](https://gallery.cortanaintelligence.com/Tutorial/Predictive-Maintenance-using-PySpark)

## How to get help and send feedback
We are eager to hear about your experience as you go through this example scenario. If you have any feedback or need help, contact us through one of these [feedback channels](../Feedback.md).

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
