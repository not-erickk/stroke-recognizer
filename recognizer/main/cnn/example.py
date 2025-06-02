#!/usr/bin/env python3
"""
Ejemplo práctico de implementación completa del sistema CNN para análisis de errores de escritura.
Integra el labeling tool existente con la nueva arquitectura CNN.

Autor: Sistema de Análisis de Escritura
Fecha: 2025
"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from keras.utils import to_categorical

# Importar clases del proyecto
from labeling_tool import LabelingSession, ERROR_CATEGORIES
from handwritting_error_cnn import HandwrittingErrorCNN, evaluate_model_performance, plot_training_history


class HandwritingAnalysisSystem:
    """
    Sistema completo de análisis de errores de escritura que integra:
    - Labeling tool existente para datos etiquetados
    - Arquitectura CNN diseñada para predicción
    - Pipeline completo de entrenamiento y evaluación
    """

    def __init__(self,
                 images_dir: str,
                 labels_csv: str = "labels.csv",
                 model_save_path: str = "handwriting_error_model.h5"):

        self.images_dir = Path(images_dir)
        self.labels_csv = Path(labels_csv)
        self.model_save_path = model_save_path

        # Inicializar componentes
        self.labeling_session = None
        self.cnn_model = None
        self.char_encoder = LabelEncoder()

        # Configuración
        self.image_size = (200, 200)
        self.char_vocab = self._build_character_vocabulary()

    def _build_character_vocabulary(self) -> list:
        """Construye vocabulario de caracteres A-Z, a-z, 0-9"""
        vocab = []
        # Letras mayúsculas
        vocab.extend([chr(i) for i in range(ord('A'), ord('Z') + 1)])
        # Letras minúsculas
        vocab.extend([chr(i) for i in range(ord('a'), ord('z') + 1)])
        # Números
        vocab.extend([str(i) for i in range(10)])
        return vocab

    def load_labeled_data(self) -> tuple:
        """
        Carga datos etiquetados del labeling tool y los prepara para CNN.

        Returns:
            tuple: (images, characters, error_scores)
        """

        print("Cargando datos etiquetados...")

        # Verificar que existe el archivo de labels
        if not self.labels_csv.exists():
            raise FileNotFoundError(f"No se encontró el archivo de labels: {self.labels_csv}")

        # Cargar labels
        labels_df = pd.read_csv(self.labels_csv)
        print(f"Encontradas {len(labels_df)} muestras etiquetadas")

        # Preparar listas para datos
        images = []
        characters = []
        error_scores = []

        # Procesar cada muestra
        for idx, row in labels_df.iterrows():

            # Cargar imagen
            img_path = self.images_dir / row['filename']
            if not img_path.exists():
                print(f"Advertencia: Imagen no encontrada {img_path}")
                continue

            # Leer y procesar imagen
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Advertencia: No se pudo leer {img_path}")
                continue

            # Redimensionar a tamaño estándar
            img_resized = self._preprocess_image(img)
            images.append(img_resized)

            # Procesar carácter
            char = row['character']
            if char not in self.char_vocab:
                print(f"Advertencia: Carácter desconocido '{char}', saltando muestra")
                continue
            characters.append(char)

            # Procesar puntuaciones de error
            scores = []
            for category in ERROR_CATEGORIES:
                if category in row:
                    scores.append(row[category])
                else:
                    print(f"Advertencia: Categoría {category} no encontrada, usando 5")
                    scores.append(5)  # Valor por defecto

            error_scores.append(scores)

        # Convertir a arrays numpy
        images = np.array(images)
        characters = np.array(characters)
        error_scores = np.array(error_scores, dtype=np.float32)

        print(f"Datos cargados exitosamente:")
        print(f"  - Imágenes: {images.shape}")
        print(f"  - Caracteres: {len(characters)} únicos")
        print(f"  - Puntuaciones: {error_scores.shape}")

        return images, characters, error_scores

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocesa una imagen individual para CNN.
        Sigue el mismo proceso que el labeling tool pero optimizado para CNN.
        """

        # Redimensionar manteniendo aspecto ratio
        height, width = img.shape
        target_size = self.image_size[0]  # 200x200

        if height > width:
            new_height = target_size
            new_width = int(width * (target_size / height))
        else:
            new_width = target_size
            new_height = int(height * (target_size / width))

        img_resized = cv2.resize(img, (new_width, new_height))

        # Centrar en canvas cuadrado
        canvas = np.ones((target_size, target_size), dtype=np.uint8) * 255
        y_offset = (target_size - new_height) // 2
        x_offset = (target_size - new_width) // 2
        canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = img_resized

        # Añadir dimensión de canal
        canvas = np.expand_dims(canvas, axis=-1)

        return canvas

    def prepare_character_encoding(self, characters: np.ndarray) -> np.ndarray:
        """
        Convierte caracteres a encoding one-hot.
        """

        # Ajustar encoder con vocabulario completo
        self.char_encoder.fit(self.char_vocab)

        # Codificar caracteres
        chars_encoded = self.char_encoder.transform(characters)

        # Convertir a one-hot
        chars_onehot = to_categorical(chars_encoded, num_classes=len(self.char_vocab))

        return chars_onehot

    def initialize_model(self):
        """Inicializa la arquitectura CNN."""

        print("Inicializando modelo CNN...")

        self.cnn_model = HandwrittingErrorCNN(
            image_size=self.image_size,
            num_error_categories=len(ERROR_CATEGORIES),
            char_vocab_size=len(self.char_vocab),
            learning_rate=0.001
        )

        # Compilar modelo
        self.cnn_model.compile_model()

        print("Modelo inicializado y compilado exitosamente")

    def train_model(self,
                    images: np.ndarray,
                    characters: np.ndarray,
                    error_scores: np.ndarray,
                    epochs: int = 100,
                    batch_size: int = 32,
                    validation_split: float = 0.2):
        """
        Entrena el modelo CNN con los datos etiquetados.
        """

        if self.cnn_model is None:
            self.initialize_model()

        print(f"Iniciando entrenamiento por {epochs} épocas...")

        # Preparar datos
        chars_encoded = self.prepare_character_encoding(characters)

        # Dividir datos
        train_data, val_data, test_data = self.cnn_model.prepare_data(
            images, chars_encoded, error_scores,
            test_size=0.15,
            val_size=validation_split
        )

        # Configurar callbacks
        callbacks = self.cnn_model.get_callbacks(
            model_checkpoint_path=self.model_save_path,
            early_stopping_patience=20
        )

        # Entrenar
        history = self.cnn_model.train(
            train_data=train_data,
            val_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks_list=callbacks
        )

        # Evaluar en test
        print("\nEvaluando en conjunto de test...")
        test_results = evaluate_model_performance(self.cnn_model, test_data)

        # Mostrar resultados
        self._print_evaluation_results(test_results)

        # Visualizar entrenamiento
        plot_training_history(history)

        return history, test_results

    def _print_evaluation_results(self, results: dict):
        """Imprime resultados de evaluación de manera organizada."""

        print("\n" + "=" * 70)
        print("RESULTADOS DE EVALUACIÓN DEL MODELO")
        print("=" * 70)

        # Tabla de resultados por categoría
        print(f"{'Categoría':<25} {'MAE':<8} {'RMSE':<8} {'Correlación':<12}")
        print("-" * 70)

        total_mae = 0
        total_rmse = 0
        total_corr = 0

        for category, metrics in results.items():
            mae = metrics['mae']
            rmse = metrics['rmse']
            corr = metrics['correlation']

            print(f"{category:<25} {mae:<8.3f} {rmse:<8.3f} {corr:<12.3f}")

            total_mae += mae
            total_rmse += rmse
            total_corr += corr

        # Promedios
        n_categories = len(results)
        print("-" * 70)
        print(
            f"{'PROMEDIO':<25} {total_mae / n_categories:<8.3f} {total_rmse / n_categories:<8.3f} {total_corr / n_categories:<12.3f}")

        # Interpretación
        avg_mae = total_mae / n_categories
        print(f"\nInterpretación:")
        print(f"  - Error promedio: {avg_mae:.3f} (en escala 0-1)")
        print(f"  - Error en escala original: {avg_mae * 10:.1f} puntos (en escala 1-10)")

        if avg_mae < 0.1:
            print("  - Rendimiento: EXCELENTE (±1 punto)")
        elif avg_mae < 0.2:
            print("  - Rendimiento: BUENO (±2 puntos)")
        elif avg_mae < 0.3:
            print("  - Rendimiento: ACEPTABLE (±3 puntos)")
        else:
            print("  - Rendimiento: NECESITA MEJORA (>±3 puntos)")

    def load_model(self, model_path: str = None):
        """Carga modelo previamente entrenado."""

        if model_path is None:
            model_path = self.model_save_path

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

        print(f"Cargando modelo desde {model_path}...")

        # Inicializar arquitectura
        self.initialize_model()

        # Cargar pesos
        self.cnn_model.model.load_weights(model_path)

        print("Modelo cargado exitosamente")

    def predict_single_image(self,
                             image_path: str,
                             character: str) -> dict:
        """
        Predice errores para una imagen individual.

        Args:
            image_path: Ruta a la imagen
            character: Carácter que se supone representa

        Returns:
            dict: Puntuaciones de error por categoría
        """

        if self.cnn_model is None:
            raise ValueError("Modelo no inicializado. Cargar modelo primero.")

        # Cargar y procesar imagen
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")

        img_processed = self._preprocess_image(img)
        img_batch = np.expand_dims(img_processed, axis=0)

        # Procesar carácter
        if character not in self.char_vocab:
            raise ValueError(f"Carácter no reconocido: {character}")

        char_encoded = self.char_encoder.transform([character])
        char_onehot = to_categorical(char_encoded, num_classes=len(self.char_vocab))

        # Predicción
        predictions = self.cnn_model.predict_errors(img_batch, char_onehot)
        scores = predictions[0]  # Primera (y única) muestra

        # Convertir a diccionario con nombres de categorías
        results = {}
        for i, category in enumerate(ERROR_CATEGORIES):
            # Convertir de escala 0-1 a 1-10
            score_1_10 = scores[i] * 9 + 1  # Mapear [0,1] -> [1,10]
            results[category] = float(score_1_10)

        return results

    def analyze_batch_images(self, images_dir: str, output_csv: str = "batch_analysis.csv"):
        """
        Analiza un lote de imágenes y guarda resultados en CSV.
        """

        images_path = Path(images_dir)
        results = []

        print(f"Analizando imágenes en {images_path}...")

        # Procesar cada imagen
        for img_file in images_path.glob("*.png"):
            try:
                # Extraer carácter del nombre de archivo (convención)
                # Ejemplo: "a_001.png" -> "a"
                char = img_file.stem.split('_')[0]

                # Predecir errores
                errors = self.predict_single_image(str(img_file), char)

                # Guardar resultado
                result = {'filename': img_file.name, 'character': char}
                result.update(errors)
                results.append(result)

            except Exception as e:
                print(f"Error procesando {img_file}: {e}")

        # Guardar resultados
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_csv, index=False)

        print(f"Análisis completado. Resultados guardados en {output_csv}")
        print(f"Procesadas {len(results)} imágenes")

        return results_df


def main():
    """Ejemplo de uso completo del sistema."""

    print("=" * 70)
    print("SISTEMA DE ANÁLISIS DE ERRORES DE ESCRITURA")
    print("=" * 70)

    # Configuración
    images_dir = "character_images"  # Directorio con imágenes etiquetadas
    labels_file = "labels.csv"  # Archivo generado por labeling tool

    # Inicializar sistema
    system = HandwritingAnalysisSystem(
        images_dir=images_dir,
        labels_csv=labels_file
    )

    try:
        # Cargar datos etiquetados
        images, characters, error_scores = system.load_labeled_data()

        # Entrenar modelo
        print("\nIniciando entrenamiento...")
        history, results = system.train_model(
            images=images,
            characters=characters,
            error_scores=error_scores,
            epochs=50,  # Reducido para demo
            batch_size=16
        )

        print("\nEntrenamiento completado exitosamente!")

        # Ejemplo de predicción individual
        if len(images) > 0:
            print("\nEjemplo de predicción individual:")
            sample_idx = 0
            sample_char = characters[sample_idx]

            # Simular guardado temporal de imagen
            sample_img_path = "temp_sample.png"
            cv2.imwrite(sample_img_path, images[sample_idx].squeeze())

            # Predecir
            predicted_errors = system.predict_single_image(sample_img_path, sample_char)

            print(f"Carácter: '{sample_char}'")
            print("Errores detectados:")
            for category, score in predicted_errors.items():
                print(f"  {category}: {score:.1f}/10")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPara usar este sistema:")
        print("1. Ejecutar el labeling tool para crear labels.csv")
        print("2. Asegurarse de que las imágenes estén en el directorio correcto")
        print("3. Ejecutar este script nuevamente")

    except Exception as e:
        print(f"Error inesperado: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()