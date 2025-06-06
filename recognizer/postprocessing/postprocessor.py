from recognizer.postprocessing.prediction_analysis import PredictionAnalysis

def analyze(objective_word: str, aw_prediction: dict):
    return PredictionAnalysis(objective_word, aw_prediction).analyze()