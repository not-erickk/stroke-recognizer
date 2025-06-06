import json

from oldpostprocessing import HandwritingMetricsAnalyzer
from prediction_analysis import PredictionAnalysis


def test_flow():
    import os
    # Test PredictionAnalysis
    test_json_path = os.path.join(os.path.dirname(__file__), "test_predictions.json")
    with open(test_json_path) as f:
        test_predictions = json.load(f)
    
    # Test with "Recámara"
    pred_analyzer = PredictionAnalysis("Recámara", test_predictions)
    result = pred_analyzer.analyze()
    
    print("PredictionAnalysis Test Results:")
    print(f"Objective word: {pred_analyzer.objective_word}")
    print(f"Analysis result: {result}")
    
    return

if __name__ == "__main__":
    # First test PredictionAnalysis
    test_flow()
    
    print("\n" + "="*50 + "\n")
    # Updated grid size based on actual letter sizes in the test data
    analyzer = HandwritingMetricsAnalyzer(
        expected_word="recamara",
        confidence_threshold=0.5,
        grid_size=(150, 150)  # More realistic based on detected letter sizes
    )

    import os
    json_path = os.path.join(os.path.dirname(__file__), "test_predictions.json")
    sample_detections = json.load(open(json_path))

    # Test with debug output
    print("Testing handwriting analysis...")
    print(f"Expected word: {analyzer.expected_word}")
    print(f"Grid size: {analyzer.grid_size}")
    
    metrics = analyzer.analyze(sample_detections)
    print("\nMetrics:", metrics)

    report = analyzer.get_detailed_report(sample_detections)
    print("\nDetailed Report:")
    for key, value in report.items():
        print(f"{key}: {value}")
        
    # Show detected letters info
    print("\nDetected letters:")
    if isinstance(sample_detections, dict) and 'predictions' in sample_detections:
        predictions = sample_detections['predictions']
    else:
        predictions = sample_detections
        
    for pred in predictions:
        if pred['confidence'] >= 0.5 and pred['class'].lower() in 'abcdefghijklmnñopqrstuvwxyz':
            print(f"  {pred['class']} at ({pred['x']:.1f}, {pred['y']:.1f}) size: {pred['width']:.1f}x{pred['height']:.1f}")