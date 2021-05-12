---
title: "5.1 Automating end-to-end ML pipelines"
weight: 3
---

{{% notice info %}}
Start this section in a new Jupyter notebook with the Data Science kernel
{{% /notice %}}

```python
import json
import time
import pathlib
import numpy as np
import pandas as pd
from time import gmtime, strftime

import boto3

import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.processing import ProcessingInput, ProcessingOutput, FeatureStoreOutput
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.workflow.parameters import ParameterInteger, ParameterFloat, ParameterString
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CreateModelStep
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

random_state = 42
```


```python
%store -r
%store
```

### Create a SageMaker Pipeline to Automate All the Steps from Data Prep to Model Deployment
Now that youve manually done each step in our machine learning workflow, you can create a pipeline which trains a new model, persists the model in SageMaker and then adds the model to the registry.

### Pipeline parameters
An important feature of SageMaker Pipelines is the ability to define the steps ahead of time, but be able to change the parameters to those steps at execution without having to re-define the pipeline. This can be achieved by using ParameterInteger, ParameterFloat or ParameterString to define a value upfront which can be modified when you call pipeline.start(parameters=parameters) later. Only certain parameters can be defined this way.


```python
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", 'imblearn'])
```


```python
train_instance_param = ParameterString(
    name="TrainingInstance",
    default_value="ml.m4.xlarge"
)

model_approval_status = ParameterString(
    name="ModelApprovalStatus",
    default_value="PendingManualApproval"
)

deploy_model_instance_type = "ml.m4.xlarge"

# this line automatically looks for the XGBoost image URI and builds an XGBoost container.
# specify the repo_version depending on your preference.
xgboost_container = sagemaker.image_uris.retrieve("xgboost", region, "1.2-1")
```

#### Step 1: Preprocess

Download the preprocessing script
```python
!wget https://raw.githubusercontent.com/shashankprasanna/sagemaker-workshop-2021/main/notebooks/04_e2e_pipeline/preprocessing.py
```

```python
s3_client.upload_file(Filename='./preprocessing.py', Bucket=default_bucket, Key=f'{prefix}/code/preprocessing.py')

create_dataset_script_uri = f's3://{default_bucket}/{prefix}/code/preprocessing.py'


create_dataset_processor = SKLearnProcessor(
    framework_version='0.23-1',
    role=sagemaker_role,
    instance_type="ml.m5.xlarge",
    instance_count=1,
    base_job_name='credit-create-dataset',
    sagemaker_session=sagemaker_session)

create_dataset_step = ProcessingStep(
    name='CreateDataset',
    processor=create_dataset_processor,
    inputs=[ProcessingInput(
                        source=s3_raw_data,
                        destination='/opt/ml/processing/input')],
    outputs=[ProcessingOutput(output_name='train_data', source='/opt/ml/processing/output/train'),
             ProcessingOutput(output_name='test_data',  source='/opt/ml/processing/output/test')],
    job_arguments=["--train-test-split-ratio", '0.8'],
    code=create_dataset_script_uri)
```

#### Step 2: Train XGBoost Model


```python
train_instance_count = 1
train_instance_type = "ml.m4.xlarge"
content_type = "text/csv"
job_name = f'XgboostTrain-' + strftime('%d-%H-%M-%S', gmtime())
training_job_output_path = f's3://{default_bucket}/{prefix}/training_jobs'
```

##### Spot training

Managed Spot Training uses Amazon EC2 Spot instance to run training jobs instead of on-demand instances. You can specify which training jobs use spot instances and a stopping condition that specifies how long Amazon SageMaker waits for a job to run using Amazon EC2 Spot instances.

This time in the pipeline, we will perform XGBoost training using Spot Instances.


```python
use_spot_instances = False
max_run = 3600
max_wait = 7200 if use_spot_instances else None
checkpoint_s3_uri = (f's3://{default_bucket}/{prefix}/checkpoints/{job_name}' if use_spot_instances
                      else None)
print("Checkpoint path:", checkpoint_s3_uri)
```


```python
!pip install sagemaker-experiments
```


```python
# construct a SageMaker estimator that calls the xgboost-container
xgboost_container = sagemaker.image_uris.retrieve("xgboost", region, "1.2-1")

xgb_estimator = sagemaker.estimator.Estimator(image_uri=xgboost_container,
                                              hyperparameters=best_job_hp,
                                              role=sagemaker.get_execution_role(),
                                              instance_count=train_instance_count,
                                              instance_type=train_instance_type,
                                              volume_size=5,  # 5 GB
                                              output_path=training_job_output_path,
                                              use_spot_instances=use_spot_instances,
                                              max_run=max_run,
                                              max_wait=max_wait,
                                              checkpoint_s3_uri=checkpoint_s3_uri
                                             )


train_step = TrainingStep(
    name=job_name,
    estimator=xgb_estimator,
    inputs={
        'train': TrainingInput(
            s3_data=create_dataset_step.properties.ProcessingOutputConfig.Outputs['train_data'].S3Output.S3Uri,
        content_type="csv")
    }
)
```

#### Step 3: Model Pre-Deployment Step


```python
model = sagemaker.model.Model(
    name='credit-default-demo-pipeline-xgboost',
    image_uri=train_step.properties.AlgorithmSpecification.TrainingImage,
    model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
    sagemaker_session=sagemaker_session,
    role=sagemaker_role
)

inputs = sagemaker.inputs.CreateModelInput(
    instance_type="ml.m4.xlarge"
)

create_model_step = CreateModelStep(
    name="ModelPreDeployment",
    model=model,
    inputs=inputs
)
```

#### Step 4: Run Bias Metrics with Clarify


```python
# clarify config
bias_report_output_path = f's3://{default_bucket}/{prefix}/clarify-output/bias'
s3_client = boto3.client('s3', region_name=region)

bias_data_config = sagemaker.clarify.DataConfig(
    s3_data_input_path=create_dataset_step.properties.ProcessingOutputConfig.Outputs['train_data'].S3Output.S3Uri,
    label='LABEL',
    dataset_type='text/csv',
    s3_output_path=bias_report_output_path)

bias_config = sagemaker.clarify.BiasConfig(
    label_values_or_threshold=[0],
    facet_name='SEX',
    facet_values_or_threshold=[1])

analysis_config = bias_data_config.get_config()
analysis_config.update(bias_config.get_config())
analysis_config["methods"] = {"pre_training_bias": {"methods": "all"}}

clarify_config_dir = pathlib.Path('config')
clarify_config_dir.mkdir(exist_ok=True)
with open(clarify_config_dir / 'analysis_config.json', 'w') as f:
    json.dump(analysis_config, f)

s3_client.upload_file(Filename='config/analysis_config.json', Bucket=default_bucket,
                      Key=f'{prefix}/clarify-config/analysis_config.json')
```


```python
# clarify processing step
clarify_processor = sagemaker.processing.Processor(
    base_job_name='fraud-detection-demo-clarify-processor',
    image_uri=sagemaker.clarify.image_uris.retrieve(framework='clarify', region=region),
    role=sagemaker.get_execution_role(),
    instance_count=1,
    instance_type='ml.c5.xlarge')

clarify_step = ProcessingStep(
    name="ClarifyProcessor",
    processor=clarify_processor,
    inputs=[
        sagemaker.processing.ProcessingInput(
            input_name="analysis_config",
            source=f's3://{default_bucket}/{prefix}/clarify-config/analysis_config.json',
            destination="/opt/ml/processing/input/config"),
        sagemaker.processing.ProcessingInput(
            input_name="dataset",
            source=create_dataset_step.properties.ProcessingOutputConfig.Outputs['train_data'].S3Output.S3Uri,
            destination="/opt/ml/processing/input/data")
    ],
    outputs=[
        sagemaker.processing.ProcessingOutput(
            source="/opt/ml/processing/output/analysis.json",
            destination=bias_report_output_path,
            output_name="analysis_result")
    ]
)
```

#### Step 5: Register Model


```python
class ModelMetrics(object):
    """Accepts model metrics parameters for conversion to request dict."""

    def __init__(
            self,
            model_statistics=None,
            model_constraints=None,
            model_data_statistics=None,
            model_data_constraints=None,
            bias=None,
            explainability=None,
    ):
        """Initialize a ``ModelMetrics`` instance and turn parameters into dict.
        Args:
            model_constraints (MetricsSource):
            model_data_constraints (MetricsSource):
            model_data_statistics (MetricsSource):
            bias (MetricsSource):
            explainability (MetricsSource):
        """
        self.model_statistics = model_statistics
        self.model_constraints = model_constraints
        self.model_data_statistics = model_data_statistics
        self.model_data_constraints = model_data_constraints
        self.bias = bias
        self.explainability = explainability

    def _to_request_dict(self):
        """Generates a request dictionary using the parameters provided to the class."""
        model_metrics_request = {}

        model_quality = {}
        if self.model_statistics is not None:
            model_quality["Statistics"] = self.model_statistics._to_request_dict()
        if self.model_constraints is not None:
            model_quality["Constraints"] = self.model_constraints._to_request_dict()
        if model_quality:
            model_metrics_request["ModelQuality"] = model_quality

        model_data_quality = {}
        if self.model_data_statistics is not None:
            model_data_quality["Statistics"] = self.model_data_statistics._to_request_dict()
        if self.model_data_constraints is not None:
            model_data_quality["Constraints"] = self.model_data_constraints._to_request_dict()
        if model_data_quality:
            model_metrics_request["ModelDataQuality"] = model_data_quality

        if self.bias is not None:
            model_metrics_request["Bias"] = {"Report": self.bias._to_request_dict()}
            # model_metrics_request["Bias"] = self.bias._to_request_dict()
        if self.explainability is not None:
            model_metrics_request["Explainability"] = self.explainability._to_request_dict()
        return model_metrics_request
```


```python
model_metrics = ModelMetrics(
    bias=sagemaker.model_metrics.MetricsSource(
        s3_uri=clarify_step.properties.ProcessingOutputConfig.Outputs['analysis_result'].S3Output.S3Uri,
        content_type="application/json"
    )
)

if 'mpg_name' not in locals():
    mpg_name = prefix
    print(f'Model Package Group name: {mpg_name}')

register_step = RegisterModel(
    name="XgboostRegisterModel",
    estimator=xgb_estimator,
    model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["text/csv"],
    response_types=["text/csv"],
    inference_instances=["ml.t2.medium", "ml.m5.xlarge"],
    transform_instances=["ml.m5.xlarge"],
    model_package_group_name=mpg_name,
    approval_status=model_approval_status,
    model_metrics=model_metrics
)
```

#### Step 6: Deploy Model

Download the deploy_model.py script
```python
!wget https://raw.githubusercontent.com/shashankprasanna/sagemaker-workshop-2021/main/notebooks/04_e2e_pipeline/deploy_model.py
```

```python
endpoint_name = "xgboost-model-pipeline-" + strftime('%d-%H-%M-%S', gmtime())
deploy_model_script_uri = f's3://{default_bucket}/{prefix}/code/deploy_model.py'


s3_client.upload_file(Filename='deploy_model.py', Bucket=default_bucket, Key=f'{prefix}/code/deploy_model.py')

deploy_model_processor = SKLearnProcessor(
    framework_version='0.23-1',
    role=sagemaker_role,
    instance_type="ml.t3.medium",
    instance_count=1,
    base_job_name='fraud-detection-demo-deploy-model',
    sagemaker_session=sagemaker_session)

deploy_step = ProcessingStep(
    name='DeployModel',
    processor=deploy_model_processor,
    job_arguments=[
        "--model-name", create_model_step.properties.ModelName,
        "--region", region,
        "--endpoint-instance-type", deploy_model_instance_type,
        "--endpoint-name", endpoint_name],
    code=deploy_model_script_uri)
```

#### Combine the Pipeline Steps and Run


```python
pipeline_name = f'credit-default'

pipeline = Pipeline(
    name=pipeline_name,
    parameters=[
        train_instance_param,
        model_approval_status],
    steps=[
        create_dataset_step,
        train_step,
        create_model_step,
        clarify_step,
        register_step,
        deploy_step
    ])
```

#### Submit the pipeline definition to the SageMaker Pipeline service

Note: If an existing pipeline has the same name it will be overwritten.


```python
pipeline.upsert(role_arn=sagemaker_role)
```

The full pipeline will take up to 30 min to run


```python
start_response = pipeline.start()

start_response.wait()
start_response.describe()
```
