---
title: "2.3 Data Preparation with SageMaker Data Wrangler"
weight: 3
---

You can also use Data Wrangler to perform a number of dataset transformations visually without writing any code. Operations include encoding categorical variables, featurizing date/time and text columns, handling missing data, manipulating columns (renaming, moving), managing rows, validating strings etc.

Let's start by launching the transform view
![](/images/dataprep/dw_transform1.png)

You should see a full list of transformations available to you on the right. We'll peform the following 3 transformations

1. Remove the ID column as it's not informative when building an ML model
1. Rename the label column from `default payment next month` to a more readable `LABEL`
1. Move the label to the first column position, since this is the format our ML algorithm expects during training

#### Drop the ID column
Click on Manage columns > Transform > Drop Column and choose `ID` under Column to drop.<br>
Click `Preview` to view your changes, and `Add` when done.
![](/images/dataprep/dw_transform2.png)
![](/images/dataprep/dw_transform3.png)

#### Rename the label column
Click on Manage columns > Transform > Rename column. Under Input column select `default payment next month` and under New name select `LABEL`.
Click preview and then save.
![](/images/dataprep/dw_transform4.png)

#### Move label column to the first position
Click on Manage columns > Transform > Move column. Under Move type select `Move to start`. Under Columns to move, select `LABEL`. Click preview and then add.
![](/images/dataprep/dw_transform5.png)

### Exporting transformations
Data Wrangler allows you to export your transformations as

1. Transformed dataset to Amazon S3
1. As Jupyter notebook that creates a data flow pipeline.
1. As Jupyter notebook that include python code for. generating the transformations using Spark.
1. As transformed features to Amazon SageMaker feature store.

To export, you'll have to first select all the steps you want to export.
![](/images/dataprep/dw_transform6.png)

And then click on `Export step` to select your export options.
![](/images/dataprep/dw_transform7.png)

{{% notice warning %}}
NOTE: The data transformations in this workshop are very basic, therefore we won't be using the output of Data Wrangler in the next section for training. <br> We will instead perform the 3 transformation steps in code, as an alternate way to do data transformations in SageMaker Studio. <br> However, feel free to click on each of the export options to take a look at the generated Jupyter notebook to perform the export steps.
{{% /notice %}}

That brings us to the end of the data preparation section. In the next section you'll learn how to build, train and tune machine learning models with Amazon SageMaker.
