---
title: "Setup Amazon SageMaker Studio"
weight: 1
chapter: false
---

## Step 1.  Log in to the Amazon SageMaker console

1. Visit the [Amazon SageMaker webpage](https://aws.amazon.com/sagemaker/?trk=el_a134p000006vgXgAAI&trkCampaign=NA-FY21-GC-400-FTSA-SAG-Overview&sc_channel=el&sc_campaign=Y21-SageMaker_shshnkp&sc_outcome=AIML_Digital_Marketing) and click on `Get Started with SageMaker` to be directed to the Amazon SageMaker console.

![](/images/sagemaker.png)

## Step 2.  Set up Amazon SageMaker Studio

In this step, you&#39;ll setup Amazon SageMaker Studio. The SageMaker Studio Notebooks within Studio are one-click Jupyter notebooks and contain everything you need to build and test your training scripts. Studio also includes experiment tracking and visualization so that it&#39;s easy to manage your entire machine learning workflow in one place.

1. Navigate to Amazon SageMaker Console
![](/images/setup/image002.png)

2. If this is your first time using Amazon SageMaker Studio, you must complete the [Studio onboarding process](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-studio-onboard.html). When onboarding, you can choose to use either [AWS Single Sign-On (AWS SSO)](https://aws.amazon.com/single-sign-on/) or [AWS Identity and Access Management (IAM)](https://aws.amazon.com/iam/) for authentication methods. When you use IAM authentication, you can choose either the Quick start or the Standard setup procedure. For simplicity, this tutorial uses the Quick start procedure.
2. In the  **Get started**  box, choose  **Quick start**. Leave the default name as it is.
![](/images/setup/image003.png)

1. Under **Execution role** , choose  **Create an IAM role**. In the dialog box that appears, choose  **Any S3 bucket**  and choose Create role. Amazon SageMaker creates a role with the required permissions and assigns it to your instance. This allows your Amazon SageMaker instance to access all the Amazon S3 buckets in your account. If you already have a bucket that you&#39;d like to use instead, select Specific S3 buckets and specify the bucket name.
![](/images/setup/image004.png)

1. Select Create role.
 Amazon SageMaker creates the _AmazonSageMaker-ExecutionRole-\*\*\*_ role.
![](/images/setup/image005.png)

1. Click **Submit.** You should see a message saying that the studio is being configured. This step should take less than 5 minutes.
![](/images/setup/image006.png)

1. When Amazon SageMaker Studio is ready, click **Open Studio**
![](/images/setup/image007.png)

1. Wait for Amazon SageMaker Studio to load. You'll see the following on your screen and it should take a few minutes launch the interface. This step should take less than 5 mins.
![](/images/setup/studio_loading.png)

1. Amazon SageMaker Studio should now be open on a separate browser tab.
![](/images/setup/image008.png)
