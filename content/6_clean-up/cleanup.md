---
title: "Delete all resources"
weight: 3
---

{{% notice info %}}
Start this section in a new Jupyter notebook with the Data Science kernel
{{% /notice %}}

After running the workshop, you can remove the resources which you created using this helper function.

First download the helper function. In a new cell execute the following.

```python
!wget https://raw.githubusercontent.com/shashankprasanna/sagemaker-workshop-2021/main/notebooks/04_e2e_pipeline/utils.py
```

```python
%store -r
```

Run the following command to delete all your project resources.
You can also delete all the objects in the project's S3 directory by passing the keyword argument `delete_s3_objects=True`.


```python
import boto3
boto_session = boto3.Session()
sagemaker_boto_client = boto_session.client('sagemaker')

from utils import delete_project_resources

delete_project_resources(
    sagemaker_boto_client=sagemaker_boto_client,
    endpoint_name=endpoint_name,
    pipeline_name='credit-default',
    mpg_name=mpg_name,
    prefix=prefix,
    delete_s3_objects=False,
    bucket_name=default_bucket)
```
