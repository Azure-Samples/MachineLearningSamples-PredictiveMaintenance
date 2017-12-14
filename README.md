# Advanced Scenario: General Predictive Maintenance

The detailed documentation for this real world scenario includes the step-by-step walk-through:
[https://docs.microsoft.com/azure/machine-learning/preview/scenario-predictive-maintenance](https://docs.microsoft.com/azure/machine-learning/preview/scenario-predictive-maintenance)

The public GitHub repository for this real world scenario contains all the code samples:
[https://github.com/Azure/MachineLearningSamples-PredictiveMaintenance](https://github.com/Azure/MachineLearningSamples-PredictiveMaintenance)

# Introduction
![](images/042116_1633_PredictiveM1.png "Predictive Maintenance")

Understanding fleet maintenance requirements can have a large impact on business safety and profitability. The business problem for this simulated data is to predict issues caused by component failures. The business question therefore is “*What is the probability that a machine goes down due to failure of a component within the next 7 days*?” This problem is formatted as a multi-class classification problem (multiple components per machine) and a machine learning algorithm is used to create the predictive model. The model is trained on historical data collected from machines. In this scenario, the user goes through the various steps of implementing such a model within the Azure Machine Learning Workbench environment.

An initial approach is to rely on **corrective maintenance**, where parts are replaced as they fail. Corrective maintenance ensures parts are used completely (not wasting component life), but incurs expense in both downtime and unscheduled maintenance requirements (off hours, or inconvenient locations).

An alternative is a **preventative maintenance** schedule. Here a business may track or test component failures and determine a safe lifespan in which to replace that component before failure. For safety critical machinery, this approach can insure no catastrophic failures. The down side is components are replaced frequently, many with remaining life left. 

The goal of **predictive maintenance** is to optimize the balance between corrective and preventative maintenance. This approach only replaces those components when they are close to failure. The savings come from both extending component lifespans (compared to preventive maintenance), and reducing unscheduled maintenance (over corrective maintenance).

The goal of this scenario is to guide a data scientist through the implementation and operationalization of the predictive maintenance solution using *Azure Machine Learning Workbench*. 

# Prerequisites

- An [Azure account](https://azure.microsoft.com/free/) (free trials are available).
- An installed copy of Azure Machine Learning Workbench with a workspace created.
- For model operationalization: Azure Machine Learning Operationalization with a local deployment environment setup and a [model management account](https://docs.microsoft.com/en-us/azure/machine-learning/preview/model-management-overview)

This example can be run on any AML Workbench compute context. However, it is recommended to run it with at least of 16-GB memory. This scenario was built and tested on a Windows 10 machine running a remote DS4_V2 standard [Data Science Virtual Machine for Linux (Ubuntu)](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/microsoft-ads.linux-data-science-vm-ubuntu).

## Login

Once you have install the AML Workbench app, we need to connect the app to your Azure subscription. From the AML Workbench `File` menu, select either the `Open Command Prompt` or `Open PowerShell` CLI. The CLI interface allows you to access your Azure services using the `az` commands. First login to your Azure account with the command:

```
az login
``` 

This will generate a key to be used with the `https:\\aka.ms\devicelogin` URL. The CLI will remain blocked until the device login operation returns.

## Create a new project

To create a new project, either use the `+` icon from the `PROJECTS` pane, or select `New Project...` from the `File` menu. The Project dialog only requires entering a Project name which is used for the directory name as well as the project name in the `PROJECTS` workbench pane. You can select a project template, such as the `Predictive Maintenance` example template. This will install the example files to explore the workbench environment.

## Connect to a remote DSVM

The predictive maintenance tutorial can be run within a local docker environment on a machine with enough memory (>=16G ram). We suggest using an Azure Linux Data Science Virtual machine (DSVM) to ensure the minimum compute resources. The scenario was developed using the DS4_V2 standard [Data Science Virtual Machine for Linux (Ubuntu)](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/microsoft-ads.linux-data-science-vm-ubuntu). 

When creating the DSVM
 * Enable the username and password connection option.
 * Enable SSH connectivity.

Once the DSVM is provisioned, we connect the AML Project to the Linux DSVM using the CLI (`File` menu, select either the `Open Command Prompt` or `Open PowerShell` CLI). 

`az ml computetarget attach remotedocker --name [Connection_Name] --address [VM_IP_Address] --username [VM_Username] --password [VM_UserPassword]`

Where:

  * [Connection_Name] is the name you'd like to use to refer to the dsvm. We use `LDSVM`, but this name is your choice.
  * [VM_IP_Address] is either the public IP address or the DNS name assigned to the DSVM in the azure portal.
  * [VM_Username] the DSVM username used during creation.
  * [VM_UserPassword] the DSVM password used during creation

Once the connection information is stored, we prepare the Docker run time environment on the DSVM using the following CLI command

`az ml experiment prepare -c [Connection_Name]`

# Let's Begin

With the docker images _prepared_, open the Jupyter notebook server either within the *AML Workbench* notebooks tab, or start a browser-based server, run:

`az ml notebook start`

The CLI command starts a local Jupyter notebook server and opens the default browser tab pointing to the project root directory. The example notebooks are stored in the `Code` directory. The predictive maintenance example runs these notebooks sequentially as numbered, starting with the Data Ingestion process in the  `Code\1_data_ingestion.ipynb` notebook. Whe you first open a notebook, the server will prompt you to connect to a kernel. Use the kernel associated with the docker container under [Project_Name]_Template [Connection_Name].

The example notebooks are broken into separate chunks of work:

 * `Code/1_data_ingestion.ipnyb` download and prepare raw data
 * `Code/2_feature_engineering.ipnyb` create model features and target label
 * `Code/3_model_building.ipnyb` build and compare machine learning model
 * `Code/4_operationlization.ipnyb` deploy a model for production scenario

 Each notebook will store intermediate results in an Azure Blob storage container to facilitate a seamless workflow. In order to do this, we require you're storage container access keys to be copied into each notebook. You can select a storage container in the https://portal.azure.com. Search for a `storage account` you'd like to use. Select the `account keys` item, and copy the `[ACCOUNT_NAME]` and one of the `[ACCOUNT_KEYS]` into the notebook code chunk: 

 ```
 # Enter your Azure blob storage details here 
ACCOUNT_NAME = "<your blob storage account name>"

# You can find the account key under the _Access Keys_ link in the 
# [Azure Portal](portal.azure.com) page for your Azure storage container.
ACCOUNT_KEY = "<your blob storage account key>"
 ``` 

Each of the four notebooks will require the same access credentials in order to load the previous intermediate results. 

## Task 1: Prepare your data

The Data Ingestion Jupyter Notebook in the `Code/1_data_ingestion.ipnyb` loads the five input data sets into `PySpark` format and does some preliminary data visualization. The data is then stored in an Azure Blob storage container on your subscription for use in the feature engineering task.

Once you have supplied you Azure storage account access keys, you can either run each cell individually, or `Run All Cells` from the `Cell` menu. This notebook will take approximately 10 minutes to run all cells.

## Task 2: Feature Engineering

Feature Engineering Jupyter Notebook in `Code/2_feature_engineering.ipnyb`, that reads `PySpark` data sets and creates the time series features used in the modeling building task. The resulting feature data set is also stored in your Azure Blob storage container.

Once you have supplied you Azure storage account access keys, you can either run each cell individually, or `Run All Cells` from the `Cell` menu. This notebook will take approximately 20 minutes to run all cells.

## Task 3: Model Building & Evaluation

The Model Building Jupyter Notebook in `Code/3_model_building.ipnyb` that reads `PySpark` feature set from blob storage and splits into the train and test data sets based on the date-timestamp. Then two models, a Decision Tree Classifier and a Random Forest Classifier, are built with the training data sets. The model performance measured on the test set is compared to determine a "best" solution to predict component failures. The resulting model is serialized and stored in the local compute context for use in the operationalization task.

Once you have supplied you Azure storage account access keys, you can either run each cell individually, or `Run All Cells` from the `Cell` menu. This notebook will take approximately 2 minutes to run all cells.

## Task 4: Operationalization

The operationalization Jupyter Notebook in `Code/4_operationalization.ipnyb` that takes the stored model and builds required functions and schema for calling the model on an Azure hosted web service. The notebook tests the functions, and zips the operationalization assets into a zip file that is also stored in your Azure Blob storage container. 

Once you have supplied you Azure storage account access keys, you can either run each cell individually, or `Run All Cells` from the `Cell` menu. This notebook will take approximately 1 minute to run all cells.

The operationalization zipped file (`o16n.zip`) contains three assets: `pdmrfull.model`, `pdmscore.py`,  `service_schema.json`. The notebook then details instructions for how to deploy this model for integration into a full predictive maintenance solution workflow. 

# Conclusion

This scenario gives the reader an overview of how to build an end to end predictive maintenance solution using PySpark within the Jupyter notebook environment in *Azure Machine Learning Workbench*. The scenario also guides the reader on how the best model can be easily operationalized and deployed using *Azure Machine Learning Model Management* environment for use in a production environment for making real time failure predictions. Then the reader can edit relevant parts of the scenario to fit their business needs.  

# Data/Telemetry
 This advance scenarios for *General Predictive Maintenance* collects usage data and sends it to Microsoft to help improve our products and services. Read our [privacy statement](https://privacy.microsoft.com/en-us/privacystatement) to learn more. 

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot automatically determines whether you need to provide a CLA and decorate the PR appropriately. You only need to follow the instructions provided by the bot across all Microsoft repository to use our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
More information is available at Code of Conduct FAQ or
contacts [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
