# Advanced Scenario: General Predictive Maintenance


The detailed documentation for this real world scenario includes the step-by-step walk-through:
[https://docs.microsoft.com/azure/machine-learning/preview/scenario-predictive-maintenance](https://docs.microsoft.com/azure/machine-learning/preview/scenario-predictive-maintenance)

The public GitHub repository for this real world scenario contains all the code samples:
[https://github.com/Azure/MachineLearningSamples-PredictiveMaintenance](https://github.com/Azure/MachineLearningSamples-PredictiveMaintenance)

## Introduction
![](images/042116_1633_PredictiveM1.png "Predictive Maintenance")

Understanding fleet maintenance requirements can have a large impact on business safety and profitability. The business problem for this simulated data is to predict issues caused by component failures. The business question therefore is “*What is the probability that a machine goes down due to failure of a component within the next 7 days*?” This problem is formatted as a multi-class classification problem (multiple components per machine) and a machine learning algorithm is used to create the predictive model. The model is trained on historical data collected from machines. In this scenario, the user goes through the various steps of implementing such a model within the Azure Machine Learning Workbench environment.

An initial approach is to rely on **corrective maintenance**, where parts are replaced as they fail. Corrective maintenance ensures parts are used completely (not wasting component life), but incurs expense in both downtime and unscheduled maintenance requirements (off hours, or inconvenient locations).

An alternative is a **preventative maintenance** schedule. Here a business may track or test component failures and determine a safe lifespan in which to replace that component before failure. For safety critical machinery, this approach can insure no catastrophic failures. The down side is components are replaced frequently, many with remaining life left. 

The goal of **predictive maintenance** is to optimize the balance between corrective and preventative maintenance. This approach only replaces those components when they are close to failure. The savings come from both extending component lifespans (compared to preventive maintenance), and reducing unscheduled maintenance (over corrective maintenance).

The goal of this scenario is to guide a data scientist through the implementation and operationalization of the predictive maintenance solution using *Azure Machine Learning Workbench*. 

## Prerequisites

- An [Azure account](https://azure.microsoft.com/free/) (free trials are available).
- An installed copy of Azure Machine Learning Workbench with a workspace created.
- For model operationalization: Azure Machine Learning Operationalization with a local deployment environment setup and a [model management account](https://docs.microsoft.com/en-us/azure/machine-learning/preview/model-management-overview)

This example can be run on any AML Workbench compute context. However, it is recommended to run it with at least of 16-GB memory. This scenario was built and tested on a Windows 10 machine running a remote DS4_V2 standard [Data Science Virtual Machine for Linux (Ubuntu)](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/microsoft-ads.linux-data-science-vm-ubuntu).

## Let's Begin

To run on your local machine, from the AML Workbench `File` menu, select either the `Open Command Prompt` or `Open PowerShell` CLI. Within the CLI windows execute the following commands:

`az ml experiment prepare --target docker --run-configuration docker`

 We suggest running on a  DS4_V2 standard [Data Science Virtual Machine for Linux (Ubuntu)](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/microsoft-ads.linux-data-science-vm-ubuntu). Once the DSVM is configured, you need to run the following two commands:

`az ml computetarget attach --name [Desired_Connection_Name] --address [VM_IP_Address] --username [VM_Username] --password [VM_UserPassword] --type remotedocker`

`az ml experiment prepare --target [Desired_Connection_Name] --run-configuration [Desired_Connection_Name]`

With the docker images _prepared_, open the jupyter notebook server either within the *AML Workbench* notebooks tab, or to start a browser-based server, run:
`az ml notebook start`

- Notebooks are stored in the `Code` directory found in the Jupyter environment. We run these notebooks sequentially as numbered, starting on (`Code\1_data_ingestion.ipynb`).

- Select the kernel to match your [Project_Name]_Template [Desired_Connection_Name] and click Set Kernel

## Task 1: Prepare your data

The Data Ingestion Jupyter Notebook in the `Code/1_data_ingestion.ipnyb` loads the five input data sets into `PySpark` format and does some preliminary data visualization. The data is then stored in an Azure Blob storage container on your subscription for use in the feature engineering task.

## Task 2: Feature Engineering

Feature Engineering Jupyter Notebook in `Code/2_feature_engineering.ipnyb`, that reads `PySpark` data sets and creates the time series features used in the modeling building task. The resulting feature data set is also stored in your Azure Blob storage container.

## Task 3: Model Building & Evaluation

The Model Building Jupyter Notebook in `Code/3_model_building.ipnyb` that reads `PySpark` feature set from blob storage and splits into the train and test data sets based on the date-timestamp. Then two models, a Decision Tree Classifier and a Random Forest Classifier, are built with the training data sets. The model performance measured on the test set is compared to determine a "best" solution to predict component failures. The resulting model is serialized and stored in the local compute context for use in the operationalization task.

## Task 4: Operationalization

The operationalization Jupyter Notebook in `Code/4_operationalization.ipnyb` that takes the stored model and builds required functions and schema for calling the model on an Azure hosted web service. The notebook tests the functions, and zips the operationalization assets into a zip file that is also stored in your Azure Blob storage container. 

The operationalization zipped file (`o16n.zip`) contains three assets: `pdmrfull.model`, `pdmscore.py`,  `service_schema.json`. The notebook then details instructions for how to deploy this model for integration into a full predictive maintenance solution workflow. 

## Conclusion

This scenario gives the reader an overview of how to build an end to end predictive maintenance solution using PySpark within the Jupyter notebook environment in *Azure Machine Learning Workbench*. The scenario also guides the reader on how the best model can be easily operationalized and deployed using *Azure Machine Learning Model Management* environment for use in a production environment for making real time failure predictions. Then the reader can edit relevant parts of the scenario to fit their business needs.  

# Data/Telemetry
 This advance scenarios for *General Predictive Maintenance* collects usage data and sends it to Microsoft to help improve our products and services. Read our [privacy statement](https://privacy.microsoft.com/en-us/privacystatement) to learn more. 

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot automatically determines whether you need to provide a CLA and decorate the PR appropriately. You only need to follow the instructions provided by the bot across all Microsoft repository to use our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
More information is available at Code of Conduct FAQ or
contacts [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
