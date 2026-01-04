from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier


class StressPredictor:

    # =====================
    # Baselines
    # =====================
    @staticmethod
    def logistic():
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500))
        ])

    @staticmethod
    def random_forest():
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=20,
            random_state=42
        )

    @staticmethod
    def xgboost():
        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        )

    # =====================
    # Tuned models
    # =====================
    @staticmethod
    def tuned_random_forest(X, y):
        param_grid = {
            "n_estimators": [200, 400],
            "max_depth": [4, 6, 8],
            "min_samples_leaf": [10, 20, 50]
        }

        rf = RandomForestClassifier(random_state=42)

        grid = GridSearchCV(
            rf,
            param_grid,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1
        )

        grid.fit(X, y)
        return grid.best_estimator_

    @staticmethod
    def tuned_xgboost(X, y):
        param_grid = {
            "n_estimators": [200, 400],
            "max_depth": [3, 4, 6],
            "learning_rate": [0.01, 0.05],
            "subsample": [0.7, 0.9]
        }

        xgb = XGBClassifier(
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        )

        grid = GridSearchCV(
            xgb,
            param_grid,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1
        )

        grid.fit(X, y)
        return grid.best_estimator_

class DirectionalPredictor:
    """Prédit la direction du marché (up/down)"""
    
    def __init__(self, model_type="logit"):
        self.model_type = model_type
        self.model = None
        
        if model_type == "logit":
            self.model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=4000, class_weight="balanced"))
            ])
        elif model_type == "xgb":
            self.model = XGBClassifier(
                n_estimators=400,
                max_depth=3,
                learning_rate=0.05,
                random_state=42
            )
    
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


class StressPredictor:
    """Prédit les périodes de stress (forte volatilité future)"""
    
    def __init__(self):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=4000, class_weight="balanced", C=1.0))
        ])
    
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
