---
title: "2.2 Using SageMaker AutoPilot to find the best models"
weight: 2
---

{{% notice tip %}}
Watch the livestream to follow along with the presenter
{{% /notice %}}

#### Click on the last icon at the bottom on the left menu
In the dropdown menu, select `Experiments and trials`
![](/images/autopilot/autopilot1.png)

#### Click `Create Experiment` to create an AutoPilot experiment
![](/images/autopilot/autopilot2.png)

#### In the AUTOPILOT EXPERIMENT SETTINGS select the following options

* Experiment name: `credit-default-example`
* Under *Connect your data*, S3 bucket name: `sagemaker-us-east-1-XXXXXXXX` (Drop down should show only 1 option)
* Dataset file name: `sagemaker-tutorial/data/dataset_unchanged.csv`
* Target: `default payment next month`
* Under *Output data location*, S3 bucket name: `sagemaker-us-east-1-XXXXXXXX` (Drop down should show only 1 option)
* Dataset directory name: type the following `sagemaker-tutorial/outputs`
* Auto deploy: **OFF**
* Under ADVANCED SETTINGS, Max candidates: 4

This will ensure that the training jobs finish in time. Increasing this number will result in larger number of models trained, and may result in models with better accuracy.

![](/images/autopilot/autopilot3.png)

#### Monitor Autopilot experiment progress

![](/images/autopilot/autopilot4.png)

#### Open candidate generation notebook

This notebook contains generated code for all the steps Autopilot is performing to find the best models. Everything is transparent and you can modify section of this notebook and run it manually.

{{% notice info %}}
Autopilot takes about 10-15 mins to generate the candidate notebooks
{{% /notice %}}

![](/images/autopilot/autopilot5.png)
![](/images/autopilot/autopilot7.png)

#### Open data exploration notebook
![](/images/autopilot/autopilot6.png)
![](/images/autopilot/autopilot8.png)

#### Once training is complete, you can see the best model based on the F1 score
![](/images/autopilot/autopilot9.png)

#### Click on Unassigned trial components
Double click on the trial component with name `documentation-XXXXXXX` for the automatically generated bias report
![](/images/autopilot/autopilot10.png)
