
# 1. Structured data prediction using Vertex AI

## Training on Cloud AI Platform
To submit to the Cloud we use [`gcloud ai-platform jobs submit training [jobname]`](https://cloud.google.com/sdk/gcloud/reference/ml-engine/jobs/submit/training) and simply specify some additional parameters for AI Platform Training Service:
> - jobname: A unique identifier for the Cloud job. We usually append system time to ensure uniqueness
> - job-dir: A GCS location to upload the Python package to
> - runtime-version: Version of TF to use.
> - python-version: Version of Python to use.
> - region: Cloud region to train in.

Below the `-- \` we add in the arguments for our `task.py` file.

```
# cf) task.py
from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
]

setup(
    name='babyweight',
    version='0.1',
    author = 'V Lakshmanan',
    author_email = 'lak@cloud.google.com',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Baby Weight prediction in Cloud ML',
    requires=[]
)
```

```
%%bash

OUTDIR=gs://${BUCKET}/babyweight/trained_model
JOBID=babyweight_$(date -u +%y%m%d_%H%M%S)

gcloud ai-platform jobs submit training ${JOBID} \
    --region=${REGION} \
    --module-name=trainer.task \
    --package-path=$(pwd)/babyweight/trainer \
    --job-dir=${OUTDIR} \
    --staging-bucket=gs://${BUCKET} \
    --master-machine-type=n1-standard-8 \
    --scale-tier=CUSTOM \
    --runtime-version=${TFVERSION} \
    --python-version=${PYTHONVERSION} \
    -- \
    --train_data_path=gs://${BUCKET}/babyweight/data/train*.csv \
    --eval_data_path=gs://${BUCKET}/babyweight/data/eval*.csv \
    --output_dir=${OUTDIR} \
    --num_epochs=10 \
    --train_examples=10000 \
    --eval_steps=100 \
    --batch_size=32 \
    --nembeds=8
```

### View the status of your job
```
Job [babyweight_211129_014008] submitted successfully.
Your job is still active. You may view the status of your job with the command

  $ gcloud ai-platform jobs describe babyweight_211129_014008

or continue streaming the logs with the command

  $ gcloud ai-platform jobs stream-logs babyweight_211129_014008
```
![](https://images.velog.io/images/findingflow/post/bd438088-8adb-4624-af36-f8d3923908c6/image.png)

![](https://images.velog.io/images/findingflow/post/574964cb-3954-4166-b365-9bff6f74eaaf/image.png)

## Check our trained model files
- check the directory structure of our outputs of our trained model in folder we exported. 
- to deploy the `saved_model.pb` within the timestamped directory as well as the variable values in the variables folder. 
- need the path of the timestamped directory so that everything within it can be found by Cloud AI Platform's model deployment service.
```
%%bash
gsutil ls gs://${BUCKET}/babyweight/trained_model
```
gs://qwiklabs-gcp-00-49bb69e45776/babyweight/trained_model/
gs://qwiklabs-gcp-00-49bb69e45776/babyweight/trained_model/20211129015352/
gs://qwiklabs-gcp-00-49bb69e45776/babyweight/trained_model/checkpoints/
```
gs://qwiklabs-gcp-00-49bb69e45776/babyweight/trained_model/
gs://qwiklabs-gcp-00-49bb69e45776/babyweight/trained_model/20211129015352/
gs://qwiklabs-gcp-00-49bb69e45776/babyweight/trained_model/checkpoints/
```
%%bash
MODEL_LOCATION=$(gsutil ls -ld -- gs://${BUCKET}/babyweight/trained_model/2* \
                 | tail -1)
gsutil ls ${MODEL_LOCATION}
```
gs://qwiklabs-gcp-00-49bb69e45776/babyweight/trained_model/20211129015352/
gs://qwiklabs-gcp-00-49bb69e45776/babyweight/trained_model/20211129015352/saved_model.pb
gs://qwiklabs-gcp-00-49bb69e45776/babyweight/trained_model/20211129015352/assets/
gs://qwiklabs-gcp-00-49bb69e45776/babyweight/trained_model/20211129015352/variables/
```
## Deploy trained model

Deploying the trained model to act as a `REST web service` is a simple gcloud call.

```
%%bash
gcloud config set ai_platform/region global
```
Updated property [ai_platform/region].

```
%%bash
MODEL_NAME="babyweight"
MODEL_VERSION="ml_on_gcp"
MODEL_LOCATION=$(gsutil ls -ld -- gs://${BUCKET}/babyweight/trained_model/2* \
                 | tail -1 | tr -d '[:space:]')
echo "Deleting and deploying $MODEL_NAME $MODEL_VERSION from $MODEL_LOCATION"
# gcloud ai-platform versions delete ${MODEL_VERSION} --model ${MODEL_NAME}
# gcloud ai-platform models delete ${MODEL_NAME}
gcloud ai-platform models create ${MODEL_NAME} --regions ${REGION}
gcloud ai-platform versions create ${MODEL_VERSION} \
    --model=${MODEL_NAME} \
    --origin=${MODEL_LOCATION} \
    --runtime-version=2.6 \
    --python-version=3.7
```
Deleting and deploying babyweight ml_on_gcp from gs://qwiklabs-gcp-00-49bb69e45776/babyweight/trained_model/20211129015352/
