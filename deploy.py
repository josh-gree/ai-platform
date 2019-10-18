import os
import logging
from subprocess import check_output, call

import click

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@click.command()
@click.option('--model_name', help='Name of Model')
@click.option('--model_version', help='Version of Model')
@click.option('--deploy_bucket', help='Bucket in which to put model')
@click.option('--model_local_location', help='Local location of the Model to deploy')
def deploy(model_name,model_version,deploy_bucket,model_local_location):
    
    extended_env={**os.environ,'CLOUDSDK_PYTHON':'python2.7'}

    logger.info('checking if bucket exists...')
    # check if bucket exists
    bucket_name = f"gs://{deploy_bucket}/"
    cmd = ['gsutil', 'ls']
    buckets = check_output(cmd,env=extended_env).decode('utf-8')
    buckets = buckets.split('\n')
    
    bucket_exists = bucket_name in buckets

    if not bucket_exists:
        # make the bucket
        logger.info('creating bucket...')
        cmd = ['gsutil','mb',bucket_name]
        call(cmd,env=extended_env)


    # check if model exists
    logger.info('checking if model exists...')
    cmd =["gcloud","ai-platform","models","list","--format=value(name)"]
    models = check_output(cmd,env=extended_env).decode('utf-8')
    
    model_exists = model_name in models
    
    if not model_exists:
        logger.info('creating model...')
        cmd = ['gcloud', 'ai-platform', 'models', 'create', model_name]
        call(cmd,env=extended_env)

    # Check if version exists
    logger.info('checking if version exists...')
    cmd =["gcloud","ai-platform","versions","list",f"--model={model_name}","--format=value(name)"]
    versions = check_output(cmd,env=extended_env).decode('utf-8')
    versions = models.split('\n')
    
    version_exists = model_version in versions
    if not version_exists:
        cmd = [
            "gcloud",
            "ai-platform",
            "versions",
            "create",
            model_version,
            f"--model={model_name}",
            f"--origin={model_local_location}",
            f"--staging-bucket={bucket_name}",
            "--runtime-version=1.14"
        ]
        out = check_output(cmd,env=extended_env).decode('utf-8')
    else:
        logger.info("Need to use a non-existent version name")


if __name__ == "__main__":
    deploy()
