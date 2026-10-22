import re
from pathlib import Path

import joblib
import pymorphy3

MODEL_PATH = Path(__file__).parent.parent.parent / "classifier" / "model.pkl"
SUB_CONFIDENCE_THRESHOLD = 0.75

_model_data = joblib.load(MODEL_PATH)
main_model = _model_data["main"]
sub_model = _model_data["sub"]
sub_to_main_mapping = _model_data["map"]

morph = pymorphy3.MorphAnalyzer()


def lemmatize(text: str) -> str:
    text = re.sub(r"[^а-яёa-z0-9\s]", " ", text.lower())
    return " ".join(morph.parse(word)[0].normal_form for word in text.split())


def classify(query: str) -> str:
    cleaned = lemmatize(query)

    main_pred = main_model.predict([cleaned])[0]

    sub_pred = sub_model.predict([cleaned])[0]
    sub_probs = sub_model.predict_proba([cleaned])
    sub_conf = sub_probs[0][list(sub_model.classes_).index(sub_pred)]

    if sub_conf > SUB_CONFIDENCE_THRESHOLD:
        main_pred = sub_to_main_mapping.get(sub_pred, main_pred)

    return f"{main_pred} / {sub_pred}"
