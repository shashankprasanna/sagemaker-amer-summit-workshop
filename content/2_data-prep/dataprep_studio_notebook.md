---
title: "2.1 Data Preparation with Amazon SageMaker Studio Notebook"
weight: 1
---

## How to run this section
To run this section of the workshop, read the instruction above each code cell, copy the code and paste it in the Jupyter notebook cell, and click run to execute the code.

![](/images/dataprep/instructions.png)


## Data preparation with SageMaker Studio Notebook

#### Import required libraries
```python
# Importing required libraries.
import pandas as pd
import numpy as np
import seaborn as sns #visualisation
import matplotlib.pyplot as plt #visualisation
%matplotlib inline
sns.set(color_codes=True)

import boto3
import sagemaker
```
#### Make sure that you are running the latest version of the Amazon SageMaker SDK

```python
# Update SageMaker SDK if necessary
if int(sagemaker.__version__.split('.')[0]) != 2:
    !pip install sagemaker==2.24.1
    print("Updating SageMakerVersion. Please restart the kernel")
else:
    print("SageMaker SDK version is good")
```

#### Save variables using `%store` to make variables persistent across notebooks and notebook restarts
Run the cell below to load any prevously created variables. You should see a print-out of the existing variables. If you don't see anything printed then it's probably the first time you are running the notebook!

```python
%store -r
%store
```
#### Create boto3 session and sagemaker session
We'll use the boto3 and sagemaker session to get the default S3 bucket to save your datasets and training jobs.

```python
boto_session = boto3.Session()
region = boto_session.region_name
print("Region = {}".format(region))

sagemaker_boto_client = boto_session.client('sagemaker')

sagemaker_session = sagemaker.session.Session(
    boto_session=boto_session,
    sagemaker_client=sagemaker_boto_client)


default_bucket = sagemaker_session.default_bucket()  # Alterantively you can use our custom bucket here.
prefix = 'sagemaker-tutorial'  # use this prefix to store all files pertaining to this workshop.
data_prefix = prefix + '/data'

%store default_bucket
%store prefix
%store data_prefix
```
#### Download the Credit card default dataset locally to `/data/` folder
```python
local_data_dir = '../data'
!mkdir $local_data_dir
!wget -O ../data/default_of_credit_card.xls  https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls
%store local_data_dir
```

#### Check if the dataset has Null values
If the result is not 0, we need to think of imputation strategies. Imputation is the process of replacing missing values with a probable value. This is often the mean, median or mode of the column, but there are also several other imputation approaches.

```python
# load data as dataframe
local_data_path = f'{local_data_dir}/default_of_credit_card.xls'

df = pd.read_excel(local_data_path, header=1)
df.head()
```

You should notice that the dataset has no missing values, when you run the command below
```python
print(f'Total number of missing values in the data: {df.isnull().sum().sum()}')
```

#### Check the dataset for gender bias

```python
# plot the bar graph customer gender
df['SEX'].value_counts(normalize=True).plot.bar()
plt.xticks([0,1], ['Male', 'Female'])
```

{{% notice info %}}
STOP: What do you see in the resulting graph? is the gender equally distributed?
{{% /notice %}}

#### Check if the dataset is imbalanced

```python
df['default payment next month'].value_counts(normalize=True).plot.bar()
plt.xticks([0,1], ['Not Default', 'Default'])
```
{{% notice info %}}
STOP: What do you see in the resulting graph? Is the dataset balanced? What is the ratio of the number of defaulters to non-defaulters? Should this affect your ML approach?
{{% /notice %}}

#### Plot the client age distribution
```python
# plot the age distribution
plt.hist(df['AGE'], bins=30)
plt.xlabel('Clients Age Distribution')
```
{{% notice info %}}
STOP: Do you see a pattern? Does a certain age group have a higher probability of defaulting vs. other?
{{% /notice %}}


### Upload data to Amazon S3 for further analysis and processing with Amazon SageMaker Data Wrangler


```python
local_raw_path = f'{local_data_dir}/dataset.csv'
df.to_csv(local_raw_path, index=False)

response = sagemaker_session.upload_data(local_raw_path,
                                         bucket=default_bucket,
                                         key_prefix=data_prefix)
print(response)

s3_raw_data = response

%store s3_raw_data
%store local_raw_path
```

{{% notice tip %}}
Save your notebook with a meaningful name by navigating to `File > Rename Notebook` and entering a name of your choice.
{{% /notice %}}
