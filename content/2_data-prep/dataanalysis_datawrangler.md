---
title: "2.2 Data Analysis with SageMaker Data Wrangler"
weight: 2
---

Amazon SageMaker Data Wrangler reduces the time it takes to aggregate and prepare data for machine learning (ML). With SageMaker Data Wrangler, you can simplify the process of data preparation and feature engineering, and complete each step of the data preparation workflow, including data selection, cleansing, exploration, and visualization from a single visual interface.

In this section we'll use Data Wrangler to generate bias reports and pre-process our training dataset.

### Launch Data Wrangler
![](/images/dataprep/launch_dw.png)
This step should take a few mins.
![](/images/dataprep/dw_loading.png)

### Import dataset into Data Wrangler
Data Wrangler can import datasets from multiple locations. For this workshop, click on Amazon S3
![](/images/dataprep/dw_import1.png)

Navigate to `S3 > sagemaker-<REGION>-ACCOUNTID > sagemaker-tutorial > data`

This is the location you uploaded your dataset in the previous section. Click on `dataset.csv` and click import
![](/images/dataprep/dw_import2.png)

You should now see an the Data Wrangler canvas.
![](/images/dataprep/dw_interface.png)

### Data Wrangler Bias Reports
You can perform a number of operations with point-and-click operations with Data Wrangler. Click on the `+` button and you'll see options to edit data types, transform your dataset, generate reports with plots and bias analysis.

Start by clicking on the `Add analysis` option
![](/images/dataprep/dw_analysis1.png)
![](/images/dataprep/dw_bias1.png)

In the new create analysis view:

1. click on `Bias Report` under Analysis type.
1. Provide a friendly name for the report.
1. Select `default payment next month` at the label columns
1. Select predicted value as 1 (credible customer)
1. Select `SEX` under column to analyze for bias
1. Select `1` under column value to analyze for bias
1. Click on `Check for bias`

![](/images/dataprep/dw_bias2.png)
![](/images/dataprep/dw_bias3.png)

#### Data Wrangler Bias report

You should see Class Imbalance (CI) score to be 0.21 and if you scroll below, you should see a message that says:

```
Positive values indicate that the advantaged group is relatively overrepresented in the dataset, the closer CI is to 1 the more advantaged the group is.
```

This means that the male group is over represented in the dataset.

{{% notice tip %}}
Try other options in the bias report, and explore other columns and bias metrics
{{% /notice %}}

After exploring bias reports, click save

#### Data Wrangler Visualizations and Summary reports
Now let's take a look at generating a summary report
Click on `Create new analysis`
![](/images/dataprep/dw_analysis2.png)

Choose table summary from the Analysis type drop down:
![](/images/dataprep/dw_analysis3.png)

You should see a table with summary statistics for all the columns in the dataset.

{{% notice tip %}}
Try other options in the Analysis Type drop down menu. Generate scatter plots and histograms to get a better sense of the dataset you're dealing with.
{{% /notice %}}

Head back to the prepare section to perform data transformation in the next section
![](/images/dataprep/dw_bias4.png)
