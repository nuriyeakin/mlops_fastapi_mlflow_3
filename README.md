- Thise project was deployed with FastApi and MLFLOW.

- In previous ML projects, models were read locally, in this project, they will be read from MLFLOW.

### 1. Pip Install Requirements

```
pip install -r requirements.txt
```

### 2. Replace some of the codes

- You may replace some code because you can read some errors due to the name of the experiment being enrolled in the volume previously.


##### 2.1 Error_1 | In train_with_mlflow.py and model_development_with_mlflow.ipynb

- Error_1: If you encounter "experiment name" error, you have to alter "experiment_name"


### 3. Run the ML model 

- Run train_with_mlflow.py

- Check the MLFlow

### 4. Deployment of the ML Model

- If the model is successful, learn the model number

- Open "main.py", learn, decide and get model from mlflow model registry


### 5. Run Uvicorn

```
uvicorn main:app --host 0.0.0.0 --port 8002
```
