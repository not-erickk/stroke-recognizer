


class PredictionAnalysis:
    def __init__(self, objective_word: str, predictions: dict):
        self.objective_word = objective_word
        self.predictions = predictions
        
        # Separate predictions by type
        self.char_predictions = []
        self.error_predictions = []
        self.accent_predictions = []
        
        for pred in predictions.get('predictions', []):
            if pred['class'] in ['sl', 'ob', 'sc', 'il']:
                self.error_predictions.append(pred)
            elif pred['class'] == 'accent':
                self.accent_predictions.append(pred)
            else:
                self.char_predictions.append(pred)
    
    def unordered_chars(self):
        """Returns True if letters in predictions match objective word but in different order"""
        detected_chars = [p['class'] for p in self.char_predictions]
        objective_chars = list(self.objective_word.lower().replace('í', 'i').replace('á', 'a'))
        
        if len(detected_chars) != len(objective_chars):
            return False
        
        return sorted(detected_chars) == sorted(objective_chars) and detected_chars != objective_chars
    
    def missing_accent(self):
        """Returns index of letter that should have accent but doesn't"""
        word_lower = self.objective_word.lower()
        
        # Check for Día
        if word_lower == 'día':
            # Find 'i' at position 1
            for i, pred in enumerate(self.char_predictions):
                if pred['class'] == 'i':
                    # Check if there's an accent with similar X coordinates
                    i_x = pred['x']
                    has_accent = any(
                        abs(accent['x'] - i_x) < 50 
                        for accent in self.accent_predictions
                    )
                    if not has_accent:
                        return i
        
        # Check for Recámara
        elif word_lower == 'recámara':
            # Find 'a' at position 3
            a_count = 0
            for i, pred in enumerate(self.char_predictions):
                if pred['class'] == 'a':
                    if a_count == 1:  # Second 'a' is at position 3
                        # Check if there's an accent with similar X coordinates
                        a_x = pred['x']
                        has_accent = any(
                            abs(accent['x'] - a_x) < 50 
                            for accent in self.accent_predictions
                        )
                        if not has_accent:
                            return i
                    a_count += 1
        
        return None
    
    def extra_accent(self):
        """Returns True if there are extra accents"""
        accent_count = len(self.accent_predictions)
        word_lower = self.objective_word.lower()
        
        if word_lower in ['día', 'recámara']:
            return accent_count > 1
        else:
            return accent_count > 0
    
    def overlapping_chars(self):
        """Returns index of overlapping characters"""
        for i in range(len(self.char_predictions)):
            for j in range(i + 1, len(self.char_predictions)):
                pred1 = self.char_predictions[i]
                pred2 = self.char_predictions[j]
                
                # Check if Y coordinates are similar
                if abs(pred1['y'] - pred2['y']) < 30:
                    # Calculate overlap in X coordinates
                    x1_start = pred1['x'] - pred1['width'] / 2
                    x1_end = pred1['x'] + pred1['width'] / 2
                    x2_start = pred2['x'] - pred2['width'] / 2
                    x2_end = pred2['x'] + pred2['width'] / 2
                    
                    # Calculate overlap
                    overlap_start = max(x1_start, x2_start)
                    overlap_end = min(x1_end, x2_end)
                    
                    if overlap_start < overlap_end:
                        overlap_width = overlap_end - overlap_start
                        # Check if overlap is more than 50% of either character
                        if (overlap_width / pred1['width'] > 0.5 or 
                            overlap_width / pred2['width'] > 0.5):
                            return i
        
        return None
    
    def wrong_char(self):
        """Returns index of wrong character when count matches but one is different"""
        detected_chars = [p['class'] for p in self.char_predictions]
        objective_chars = list(self.objective_word.lower())
        
        # Handle accented characters for comparison
        objective_chars_normalized = [
            'i' if c == 'í' else 'a' if c == 'á' else c
            for c in objective_chars
        ]
        
        if len(detected_chars) != len(objective_chars_normalized):
            return None
        
        # Count differences
        differences = []
        for i, (detected, expected) in enumerate(zip(detected_chars, objective_chars_normalized)):
            if detected != expected:
                differences.append(i)
        
        # Return index only if exactly one character is different
        if len(differences) == 1:
            return differences[0]
        
        return None
    
    def extra_char(self):
        """Returns index of extra character"""
        detected_chars = [p['class'] for p in self.char_predictions]
        objective_chars = list(self.objective_word.lower())
        
        # Handle accented characters
        objective_chars_normalized = [
            'i' if c == 'í' else 'a' if c == 'á' else c
            for c in objective_chars
        ]
        
        if len(detected_chars) != len(objective_chars_normalized) + 1:
            return None
        
        # Try removing each character and see if it matches
        for i in range(len(detected_chars)):
            test_chars = detected_chars[:i] + detected_chars[i+1:]
            if test_chars == objective_chars_normalized:
                return i
        
        return None
    
    def incomplete_loop(self):
        """Returns index of incomplete loop error"""
        for i, pred in enumerate(self.error_predictions):
            if pred['class'] == 'il':
                return i
        return None
    
    def shrinking(self):
        """Returns index of shrinking error"""
        for i, pred in enumerate(self.error_predictions):
            if pred['class'] == 'sc':
                return i
        return None
    
    def out_of_bounds(self):
        """Returns index of out of bounds error"""
        for i, pred in enumerate(self.error_predictions):
            if pred['class'] == 'ob':
                return i
        return None
    
    def separation_from_line(self):
        """Returns index of separation from line error"""
        for i, pred in enumerate(self.error_predictions):
            if pred['class'] == 'sl':
                return i
        return None
    
    def analyze(self):
        """Runs all analysis methods and returns results"""
        results = []
        
        # Character analysis methods
        if self.unordered_chars():
            results.append({
                "type": "unordered_chars",
                "description": "Caracteres desordenados"
            })
        
        missing_idx = self.missing_accent()
        if missing_idx is not None:
            results.append({
                "type": "missing_accent",
                "description": "Acento faltante",
                "predictionIdx": missing_idx
            })
        
        if self.extra_accent():
            results.append({
                "type": "extra_accent",
                "description": "Acento extra"
            })
        
        overlap_idx = self.overlapping_chars()
        if overlap_idx is not None:
            results.append({
                "type": "overlapping_chars",
                "description": "Caracteres encimados",
                "predictionIdx": overlap_idx
            })
        
        wrong_idx = self.wrong_char()
        if wrong_idx is not None:
            results.append({
                "type": "wrong_char",
                "description": "Caracter incorrecto",
                "predictionIdx": wrong_idx
            })
        
        extra_idx = self.extra_char()
        if extra_idx is not None:
            results.append({
                "type": "extra_char",
                "description": "Caracter extra",
                "predictionIdx": extra_idx
            })
        
        # Error detection methods
        il_idx = self.incomplete_loop()
        if il_idx is not None:
            results.append({
                "type": "incomplete_loop",
                "description": "Bucle incompleto",
                "predictionIdx": il_idx
            })
        
        shrink_idx = self.shrinking()
        if shrink_idx is not None:
            results.append({
                "type": "shrinking",
                "description": "Encogimiento",
                "predictionIdx": shrink_idx
            })
        
        ob_idx = self.out_of_bounds()
        if ob_idx is not None:
            results.append({
                "type": "out_of_bounds",
                "description": "Fuera de límites",
                "predictionIdx": ob_idx
            })
        
        sl_idx = self.separation_from_line()
        if sl_idx is not None:
            results.append({
                "type": "separation_from_line",
                "description": "Separación de la línea",
                "predictionIdx": sl_idx
            })

        return results