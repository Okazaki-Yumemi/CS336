import fasttext

quality_model = fasttext.load_model(
    "data/quality/quality_classifier.bin"
)

def classify_quality(text: str) -> tuple[str, float]:
    text = " ".join(text.split())

    labels, scores = quality_model.predict(text)

    label = labels[0].replace("__label__", "") #type: ignore
    score = float(scores[0])

    return label, score