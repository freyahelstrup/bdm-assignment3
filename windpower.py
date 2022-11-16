import pandas as pd
import mlflow

## NOTE: Optionally, you can use the public tracking server.  Do not use it for data you cannot afford to lose. See note in assignment text. If you leave this line as a comment, mlflow will save the runs to your local filesystem.
# mlflow.set_tracking_uri("http://training.itu.dk:5000/")

# Set the experiment name
#mlflow.set_experiment("fhel - Wind Power")

# Import some of the sklearn modules you are likely to use.
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_graphviz
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set default values
import sys
max_depth = int(sys.argv[1]) if len(sys.argv) > 1 else 6

# Start a run
# TODO: Set a descriptive name. This is optional, but makes it easier to keep track of your runs.
with mlflow.start_run():
  # Insert path to dataset
  df = pd.read_json("dataset.json", orient="split")

  # Handle missing data
  df.dropna(how='any',inplace=True)

  # Create pipeline
  import custom_transformers as ct 
  preprocessor = ColumnTransformer(
    transformers = [
      (
        'direction',
        make_pipeline(
          ct.DirectionTransformer(['Direction']),
          StandardScaler()
        ),
        ['Direction']
      ),
      (
        'numerical',
        StandardScaler(),
        ['Speed']
      )
  ])
  regressor = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
  pipeline = make_pipeline(preprocessor, regressor)

  # TODO: Currently the only metric is MAE. You should add more. What other metrics could you use? Why?
  metrics = [
    ("MAE", mean_absolute_error, []),
    ("MSE", mean_squared_error, []),
    ("R2", r2_score, []),
  ]

  X = df[["Speed","Direction"]]
  y = df["Total"]

  number_of_splits = 5

  # Log your parameters. What parameters are important to log?
  #HINT: You can get access to the transformers in your pipeline using `pipeline.steps`
  mlflow.log_param('max_depth_used',pipeline.steps[1][1].max_depth)

  from matplotlib import pyplot as plt 
  figure = plt.figure(figsize=(10,3))
  plt.grid()

  for train, test in TimeSeriesSplit(number_of_splits).split(X,y):
    pipeline.fit(X.iloc[train],y.iloc[train])
    predictions = pipeline.predict(X.iloc[test])
    truth = y.iloc[test]

    plt.plot(truth.index, truth.values, color='black', label="Actual")
    plt.plot(truth.index, predictions, color='red', linestyle='dashed', label="Predicted")

    # Calculate and save the metrics for this fold
    for name, func, scores in metrics:
      score = func(truth, predictions)
      scores.append(score)

  plt.legend()
  plt.savefig(f"predictions.png")
  mlflow.log_artifact(f"predictions.png")

  # Log a summary of the metrics
  for name, _, scores in metrics:
    # NOTE: Here we just log the mean of the scores. 
    # Are there other summarizations that could be interesting?
    mean_score = sum(scores)/number_of_splits
    mlflow.log_metric(f"mean_{name}", mean_score)
      
  mlflow.sklearn.log_model(regressor, "model")
