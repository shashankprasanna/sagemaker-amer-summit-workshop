---
title: "4.1 Deploy XGBoost model"
weight: 1
---

{{% notice info %}}
Start this section in a new Jupyter notebook with the Data Science kernel
{{% /notice %}}


### Import necessary packages

```python
import json
import time
import s3fs
import boto3
import pandas as pd

import sagemaker
from sagemaker.s3 import S3Downloader
from sagemaker.model_monitor.dataset_format import DatasetFormat
from sagemaker.model_monitor import DataCaptureConfig, DefaultModelMonitor
```


```python
# Set region, boto3 and SageMaker SDK variables¶

#You can change this to a region of your choice
region = sagemaker.Session().boto_region_name
print("Using AWS Region: {}".format(region))

boto3.setup_default_session(region_name=region)
boto_session = boto3.Session(region_name=region)

s3_client = boto3.client('s3', region_name=region)
sagemaker_boto_client = boto_session.client('sagemaker')

sagemaker_session = sagemaker.session.Session(
    boto_session=boto_session,
    sagemaker_client=sagemaker_boto_client)

sagemaker_role = sagemaker.get_execution_role()
account_id = boto3.client('sts').get_caller_identity()["Account"]

s3 = s3fs.S3FileSystem(anon=False)

random_state = 42
```


```python
%store -r
%store
```

### Approve the Second model


```python
second_model_package = sagemaker_boto_client.list_model_packages(ModelPackageGroupName=mpg_name)['ModelPackageSummaryList'][0]
model_package_update = {
    'ModelPackageArn': second_model_package['ModelPackageArn'],
    'ModelApprovalStatus': 'Approved'
}

update_response = sagemaker_boto_client.update_model_package(**model_package_update)
```

### Create an endpoint config + endpoint


```python
approved_model_name = model_2_name

endpoint_name = f'{approved_model_name}-endpoint'
endpoint_instance_count = 1
endpoint_instance_type = "ml.m4.xlarge"


primary_container = {'ModelPackageName': second_model_package['ModelPackageArn']}
endpoint_config_name=f'{approved_model_name}-endpoint-config'
existing_configs = len(sagemaker_boto_client.list_endpoint_configs(NameContains=endpoint_config_name, MaxResults = 30)['EndpointConfigs'])

if existing_configs == 0:
    create_ep_config_response = sagemaker_boto_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[{
            'InstanceType': endpoint_instance_type,
            'InitialVariantWeight': 1,
            'InitialInstanceCount': endpoint_instance_count,
            'ModelName': approved_model_name,
            'VariantName': 'AllTraffic',

        }]
    )
%store endpoint_config_name
%store approved_model_name
```


```python
existing_endpoints = sagemaker_boto_client.list_endpoints(NameContains=endpoint_name, MaxResults = 30)['Endpoints']
if not existing_endpoints:
    create_endpoint_response = sagemaker_boto_client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name)
    %store endpoint_name

endpoint_info = sagemaker_boto_client.describe_endpoint(EndpointName=endpoint_name)
endpoint_status = endpoint_info['EndpointStatus']

while endpoint_status == 'Creating':
    endpoint_info = sagemaker_boto_client.describe_endpoint(EndpointName=endpoint_name)
    endpoint_status = endpoint_info['EndpointStatus']
    print('Endpoint status:', endpoint_status)
    if endpoint_status == 'Creating':
        time.sleep(60)
```

{{% notice info %}}
This step will take several minutes.
{{% /notice %}}


### Create a predictor


```python
predictor = sagemaker.predictor.Predictor(
    endpoint_name=endpoint_name,
    sagemaker_session=sagemaker_session)
```

### Make predictions

Sample a row from the test data


```python
s3.download(test_data_uri, f'{local_data_dir}/test.csv')
```


```python
df_test = pd.read_csv(f'{local_data_dir}/test.csv', header=None)
df_test.columns = header

df_test = df_test.drop(columns=['LABEL'])
```


```python
test_input = ','.join([str(x) for x in df_test.sample(1).values.flatten().tolist()])
```


```python
results = predictor.predict(test_input, initial_args = {"ContentType": "text/csv"})
prediction = json.loads(results)
prediction
print (f'Probablitity of default is:', prediction)
```

### Model Monitor

#### Enable real-time inference data capture

To enable data capture for monitoring the model data quality, you specify the new capture option called `DataCaptureConfig`. You can capture the request payload, the response payload or both with this configuration. The capture config applies to all variants. Please provide the Endpoint name in the following cell:


```python
s3_capture_upload_path = f's3://{default_bucket}/{prefix}/model_monitor'

# captuire option for model monitor
data_capture_config = DataCaptureConfig(
                enable_capture=True,
                sampling_percentage=100,
                destination_s3_uri= s3_capture_upload_path,
                capture_options=["REQUEST", "RESPONSE"],
                csv_content_types=["text/csv"],
                json_content_types=["application/json"]
            )
```


```python
predictor.update_data_capture_config(data_capture_config=data_capture_config)
sagemaker_session.wait_for_endpoint(endpoint=endpoint_name)
```

The contents of the single captured file should be all the data captured in an Amazon SageMaker-specific JSON-line formatted file. Each inference request is captured in a single line in the jsonl file. The line contains both the input and output merged together.

#### Baselining and continuous monitoring

In addition to collecting the data, Amazon SageMaker provides the capability for you to monitor and evaluate the data observed by the endpoints. Two tasks are needed for this:

* Create a baseline with which you compare the realtime traffic.
* Setup a schedule to continuously evaluate and compare against the baseline after it has been created.

The training dataset with which you trained the model is usually a good baseline dataset. Note that the training dataset's data schema and the inference dataset schema should exactly match (i.e. number and order of the features).

Using our training dataset, we'll ask SageMaker to suggest a set of baseline constraints and generate descriptive statistics to explore the data.


```python
baseline_data_uri = train_res_data_header_uri
baseline_results_uri = f's3://{default_bucket}/{prefix}/model_monitor/baseline'

print('Baseline data uri: {}'.format(baseline_data_uri))
print('Baseline results uri: {}'.format(baseline_results_uri))
```


```python
my_monitor = DefaultModelMonitor(
    role=sagemaker_role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    volume_size_in_gb=5,
    max_runtime_in_seconds=3600,
)

my_monitor.suggest_baseline(
    baseline_dataset=baseline_data_uri,
    dataset_format=DatasetFormat.csv(header=True),
    output_s3_uri=baseline_results_uri,

)
```

#### Explore the generated constraints and statistics

With the monitor object, you can also explore the generated constraints and statistics:


```python
baseline_job = my_monitor.latest_baselining_job
schema_df = pd.io.json.json_normalize(baseline_job.baseline_statistics().body_dict["features"])
schema_df.head(10)

constraints_df = pd.io.json.json_normalize(baseline_job.suggested_constraints().body_dict["features"])
constraints_df.head(10)
```

You can also analyze and monitor the data with Monitoring Schedules.

Using `DefaultMonitor.create_monitoring_schedule()`, you can create a model monitoring schedule for an endpoint that compares the baseline resources (constraints and statistics) against the realtime traffic. For more about this method, see the [API documentation](https://sagemaker.readthedocs.io/en/stable/model_monitor.html#sagemaker.model_monitor.model_monitoring.DefaultModelMonitor.create_monitoring_schedule).
