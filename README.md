# ai-platform

code to locally train and deploy tf estimator for image classification.

## Training

```bash
pipenv run python model/model/main.py --batch_size 50 --epochs 4 --num_hidden_units 500 
--model_name mobilenet_256
```

## Deployment

```bash
pipenv run python deploy.py --model_name model_name --model_version v1 --deploy_bucket 
deploy-model --model_local_location path/to/exported/model
```
