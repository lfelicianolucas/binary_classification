import itertools
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils import compute_class_weight


def _concat_name(name: str, array):
    return [f"{name}: {v!r}" for v in array]


class TrainingPreProcessor:
    def __init__(self) -> None:
        self.target: str = None
        self.numeric_features: List[str] = None
        self.categoric_features: List[str] = None

        self.numeric_standard_scaler = StandardScaler()
        self.categoric_one_hot_encoder = OneHotEncoder()
        self.label_encoder = LabelEncoder()

        self.class_weight = None
        self.feature_labels = None

    def fit(self, df: pd.DataFrame, ignore_columns: List[str] = None) -> None:
        if ignore_columns:
            df = df.copy().drop(columns=ignore_columns)

        self.target = "Attrition_Flag"
        self.numeric_features = [
            col
            for col in df.columns
            if pd.api.types.is_numeric_dtype(df[col]) and col != self.target
        ]
        self.categoric_features = [
            col
            for col in df.columns
            if (not pd.api.types.is_numeric_dtype(df[col])) and col != self.target
        ]

        self.numeric_standard_scaler = self.numeric_standard_scaler.fit(
            df[self.numeric_features]
        )
        self.categoric_one_hot_encoder = self.categoric_one_hot_encoder.fit(
            df[self.categoric_features]
        )
        self.label_encoder = self.label_encoder.fit(df[self.target])

        weights = compute_class_weight(
            "balanced", classes=self.label_encoder.classes_, y=df[self.target]
        )
        classes = self.label_encoder.transform(self.label_encoder.classes_)
        self.class_weight = dict(zip(classes, weights))

        self.feature_labels = dict(enumerate(self.numeric_features))
        self.feature_labels.update(
            dict(
                enumerate(
                    itertools.chain(
                        *[
                            _concat_name(cat, values)
                            for (cat, values) in zip(
                                self.categoric_features,
                                self.categoric_one_hot_encoder.categories_,
                            )
                        ]
                    ),
                    start=len(self.numeric_features),
                )
            )
        )

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        categoric_features = self.categoric_one_hot_encoder.transform(
            df[self.categoric_features]
        ).toarray()
        numeric_features = self.numeric_standard_scaler.transform(
            df[self.numeric_features]
        )
        target = self.label_encoder.transform(df[self.target])

        return np.hstack((numeric_features, categoric_features)), target
