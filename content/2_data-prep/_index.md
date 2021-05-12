---
title: "2. Data Preparation"
chapter: false
weight: 3
---

In this section you'll learn how to use Amazon SageMaker Studio and Data Wrangler to analyze your dataset.


**Dataset overview**

You'll be working with a tabular dataset consisting of default payments of credit card clients.
<br> **Number of rows:** 30000 <br> **Number of columns:** 24

Dataset source: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients

The output or response variable we'll be building a model to predict is a binary categorical variable **1 or 0**. Where **1** represents a credible customer, who will not default on the payments and **0:** represents a not very credible customer, who will default on the payments. The machine learning problem at hand is to estimate the probability of default.

The screenshot below shows part of the dataset. The red box shows the response variable that we'll train a model to predict.
![](/images/setup/dataset_csv.png)
