from sklearn.base import TransformerMixin, BaseEstimator

class DirectionTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, columnNames):
    self.columnNames = columnNames

  def fit(self, X, y = None):
    return self    
  
  def transform(self, X, y = None):
    direction_switcher = {
      "N": 90.0 * 0.0,
      "NNE": 90.0 * 0.25,
      "NE": 90.0 * 0.5,
      "ENE": 90.0 * 0.75,
      "E": 90.0 * 1.0,
      "ESE": 90.0 * 1.25,
      "SE": 90.0 * 1.5,
      "SSE": 90.0 * 1.75,
      "S": 90.0 * 2.0,
      "SSW": 90.0 * 2.25,
      "SW": 90.0 * 2.5,
      "WSW": 90.0 * 2.75,
      "W": 90.0 * 3.0,
      "WNW": 90.0 * 3.25,
      "NW": 90.0 * 3.5,
      "NNW": 90.0 * 3.75
    }

    X_ = X.copy()
    for column in self.columnNames:
      X_[column] = X_[column].apply(lambda direction : direction_switcher.get(direction, -1.0))
      
    return X_
    