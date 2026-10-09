from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass

@dataclass
class ModelBundle:
    classifier: Pipeline
    regressor: Pipeline
    feature_columns: list

def make_model_bundle(feature_columns: list, random_state: int = 42) -> ModelBundle:
    scaler = StandardScaler()
    clf = HistGradientBoostingClassifier(random_state=random_state, max_iter=300)
    reg = HistGradientBoostingRegressor(random_state=random_state, max_iter=300)
    clf_pipe = Pipeline([("scaler", scaler), ("clf", clf)])
    reg_pipe = Pipeline([("scaler", scaler), ("reg", reg)])
    return ModelBundle(classifier=clf_pipe, regressor=reg_pipe, feature_columns=feature_columns)
