---
title: "2.1 Data Preparation with Amazon SageMaker Studio Notebook"
weight: 1
---

{{% notice tip %}}
Watch the livestream to follow along with the presenter
{{% /notice %}}

### Open the notebook named `1_upload_dataset_for_autopilot.ipynb`
![](/images/setup/setup14.png)

{{% notice info %}}
A copy of the code from the notebook is also available below, if you prefer building your notebook from scratch by copy pasting each code cell and then running them.
{{% /notice %}}

#### Visualize and upload dataset to Amazon S3

#### Import packages


```python
import boto3
import sagemaker
import pandas as pd
import matplotlib.pyplot as plt
```

#### Create a sagemaker_session


```python
boto_session = boto3.Session()
sagemaker_boto_client = boto_session.client('sagemaker')

sagemaker_session = sagemaker.session.Session(
    boto_session=boto_session,
    sagemaker_client=sagemaker_boto_client)
```

#### Use the default Amazon S3 bucket for dataset and results


```python
default_bucket = sagemaker_session.default_bucket()  # Alternatively you can use your custom bucket here.

prefix = 'sagemaker-tutorial'  # use this prefix to store all files pertaining to this workshop.
data_prefix = prefix + '/data'
```

#### Visualize the dataset


```python
local_data_dir = './data'
df = pd.read_excel('./data/default_of_credit_card.xls', header=1)
df.head()
```


```python
print(f'Total number of missing values in the data: {df.isnull().sum().sum()}')
```


```python
# plot the bar graph customer gender
df['SEX'].value_counts(normalize=True).plot.bar()
plt.xticks([0,1], ['Male', 'Female'])
```


```python
# plot the age distribution
plt.hist(df['AGE'], bins=30)
plt.xlabel('Clients Age Distribution')
```
### Upload the dataset to Amazon S3

```python
df.to_csv('./data/dataset_unchanged.csv', index=False)

response = sagemaker_session.upload_data(f'{local_data_dir}/dataset_unchanged.csv',
                                         bucket=default_bucket,
                                         key_prefix=data_prefix)
print(response)
```
