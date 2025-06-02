#!/usr/bin/env python3
"""
Arquitectura CNN especializada para detección de errores en trazos de escritura.
Diseñada para analizar imágenes binarizadas de caracteres individuales e identificar
patrones específicos de errores de escritura.

Autor: Sistema de Análisis de Escritura
Fecha: 2025
"""

import tensorflow as tf
import keras
from keras import layers, models, optimizers, callbacks
from keras.utils import to_categorical
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Categorías de errores según el labeling tool
ERROR_CATEGORIES = [
    "incomplete_strokes",  # Trazos incompletos
    "incorrect_letter",  # Forma de letra incorrecta
    "dysgraphia",  # Problemas de control motor
    "size_inconsistency",  # Inconsistencia de tamaño
    "spacing_issues",  # Problemas de espaciado
    "line_alignment",  # Alineación con la línea
    "slant_variation",  # Variación de inclinación
    "pressure_irregularity",  # Presión irregular
    "proportion_errors",  # Errores de proporción
    "closure_problems"  # Problemas de cierre
]


class HandwrittingErrorCNN:
    """
    Arquitectura CNN híbrida para análisis de errores de escritura.

    Características principales:
    - Input: Imagen 200x200 + carácter esperado (one-hot encoded)
    - Output: 10 valores de regresión (0-1) para cada categoría de error
    - Arquitectura híbrida CNN + MLP para multi-output regression
    """

    def __init__(self,
                 image_size: Tuple[int, int] = (200, 200),
                 num_error_categories: int = 10,
                 char_vocab_size: int = 62,  # A-Z, a-z, 0-9
                 learning_rate: float = 0.001):

        self.image_size = image_size
        self.num_error_categories = num_error_categories
        self.char_vocab_size = char_vocab_size
        self.learning_rate = learning_rate

        # Construir el modelo
        self.model = self._build_model()

    def _build_model(self) -> keras.Model:
        """
        Construye la arquitectura CNN híbrida.

        Diseño inspirado en investigaciones de CNNs para análisis de disgrafía
        con adaptaciones para regresión multi-output.
        """

        # =================================================================
        # RAMA 1: PROCESAMIENTO DE IMAGEN (CNN)
        # =================================================================

        # Input de imagen (200x200x1 para imágenes binarizadas)
        image_input = layers.Input(
            shape=(*self.image_size, 1),
            name='image_input'
        )

        # Preprocessing: normalización
        x = layers.Rescaling(1. / 255, name='rescaling')(image_input)

        # Bloque Convolucional 1: Detección de características básicas
        # Filtros pequeños (3x3) para detectar trazos fundamentales
        x = layers.Conv2D(32, (3, 3), activation='relu',
                          padding='same', name='conv1_1')(x)
        x = layers.Conv2D(32, (3, 3), activation='relu',
                          padding='same', name='conv1_2')(x)
        x = layers.MaxPooling2D((2, 2), name='pool1')(x)
        x = layers.Dropout(0.15, name='dropout1')(x)

        # Bloque Convolucional 2: Patrones de trazos más complejos
        x = layers.Conv2D(64, (3, 3), activation='relu',
                          padding='same', name='conv2_1')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu',
                          padding='same', name='conv2_2')(x)
        x = layers.MaxPooling2D((2, 2), name='pool2')(x)
        x = layers.Dropout(0.2, name='dropout2')(x)

        # Bloque Convolucional 3: Características de nivel medio
        # Importante para detectar proporciones y estructuras
        x = layers.Conv2D(128, (3, 3), activation='relu',
                          padding='same', name='conv3_1')(x)
        x = layers.Conv2D(128, (3, 3), activation='relu',
                          padding='same', name='conv3_2')(x)
        x = layers.MaxPooling2D((2, 2), name='pool3')(x)
        x = layers.Dropout(0.25, name='dropout3')(x)

        # Bloque Convolucional 4: Características de alto nivel
        # Para detectar patrones complejos de disgrafía
        x = layers.Conv2D(256, (3, 3), activation='relu',
                          padding='same', name='conv4_1')(x)
        x = layers.Conv2D(256, (3, 3), activation='relu',
                          padding='same', name='conv4_2')(x)
        x = layers.MaxPooling2D((2, 2), name='pool4')(x)
        x = layers.Dropout(0.3, name='dropout4')(x)

        # Bloque final de características específicas
        x = layers.Conv2D(512, (3, 3), activation='relu',
                          padding='same', name='conv5')(x)
        x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)

        # =================================================================
        # RAMA 2: PROCESAMIENTO DE CARÁCTER ESPERADO
        # =================================================================

        # Input del carácter esperado (one-hot encoded)
        char_input = layers.Input(
            shape=(self.char_vocab_size,),
            name='char_input'
        )

        # Embedding denso para características del carácter
        char_dense = layers.Dense(64, activation='relu',
                                  name='char_embedding')(char_input)
        char_dense = layers.Dropout(0.2, name='char_dropout')(char_dense)

        # =================================================================
        # FUSIÓN Y CAPAS DE DECISIÓN
        # =================================================================

        # Concatenar características de imagen y carácter
        combined = layers.Concatenate(name='feature_fusion')([x, char_dense])

        # Capas densas para análisis conjunto
        combined = layers.Dense(512, activation='relu',
                                name='fusion_dense1')(combined)
        combined = layers.Dropout(0.4, name='fusion_dropout1')(combined)

        combined = layers.Dense(256, activation='relu',
                                name='fusion_dense2')(combined)
        combined = layers.Dropout(0.3, name='fusion_dropout2')(combined)

        # =================================================================
        # SALIDAS ESPECIALIZADAS POR CATEGORÍA DE ERROR
        # =================================================================

        # Crear una salida específica para cada categoría de error
        # Esto permite al modelo especializarse en detectar cada tipo
        error_outputs = []

        for i, category in enumerate(ERROR_CATEGORIES):
            # Rama específica para cada tipo de error
            error_branch = layers.Dense(128, activation='relu',
                                        name=f'{category}_branch')(combined)
            error_branch = layers.Dropout(0.2,
                                          name=f'{category}_dropout')(error_branch)

            # Salida con activación sigmoid para valores 0-1
            error_output = layers.Dense(1, activation='sigmoid',
                                        name=f'{category}_output')(error_branch)
            error_outputs.append(error_output)

        # Concatenar todas las salidas
        final_output = layers.Concatenate(name='error_scores')(error_outputs)

        # =================================================================
        # CONSTRUCCIÓN DEL MODELO
        # =================================================================

        model = keras.Model(
            inputs=[image_input, char_input],
            outputs=final_output,
            name='HandwritingErrorCNN'
        )

        return model

    def compile_model(self,
                      loss_weights: Dict[str, float] = None,
                      metrics: List[str] = None):
        """
        Compila el modelo con configuraciones optimizadas para análisis de errores.

        Args:
            loss_weights: Pesos para balancear la importancia de diferentes errores
            metrics: Métricas adicionales a monitorear
        """

        # Optimizador Adam con learning rate adaptativo
        optimizer = optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )

        # Loss function para regresión multi-output
        # MSE es apropiado para predecir valores continuos 0-1
        loss = 'mse'

        # Métricas por defecto
        if metrics is None:
            metrics = ['mae', 'mse']

        # Compilar modelo
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

        print("Modelo compilado exitosamente")
        print(f"Parámetros totales: {self.model.count_params():,}")

    def get_callbacks(self,
                      model_checkpoint_path: str = 'best_model.h5',
                      early_stopping_patience: int = 15) -> List[callbacks.Callback]:
        """
        Configura callbacks para entrenamiento optimizado.
        """

        callback_list = [
            # Reducir learning rate cuando plateau
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            ),

            # Early stopping para evitar overfitting
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),

            # Guardar mejor modelo
            callbacks.ModelCheckpoint(
                filepath=model_checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),

            # Logging de progreso
            callbacks.CSVLogger('training_log.csv', append=True)
        ]

        return callback_list

    def prepare_data(self,
                     images: np.ndarray,
                     characters: np.ndarray,
                     error_scores: np.ndarray,
                     test_size: float = 0.2,
                     val_size: float = 0.15) -> Tuple:
        """
        Prepara los datos para entrenamiento.

        Args:
            images: Array de imágenes (N, 200, 200, 1)
            characters: Array de caracteres (one-hot o labels)
            error_scores: Array de puntuaciones de error (N, 10)
            test_size: Proporción para test
            val_size: Proporción para validación

        Returns:
            Tuple con datos divididos
        """

        # Normalizar puntuaciones de error a rango 0-1
        error_scores_norm = error_scores / 10.0  # Asumiendo escala 1-10

        # Asegurar que las imágenes tengan la forma correcta
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=-1)

        # Codificar caracteres si es necesario
        if characters.shape[-1] != self.char_vocab_size:
            # Asumir que son labels, convertir a one-hot
            char_encoder = LabelEncoder()
            chars_encoded = char_encoder.fit_transform(characters)
            characters_onehot = to_categorical(chars_encoded,
                                               num_classes=self.char_vocab_size)
        else:
            characters_onehot = characters

        # División train/test
        X_img_temp, X_img_test, X_char_temp, X_char_test, y_temp, y_test = \
            train_test_split(images, characters_onehot, error_scores_norm,
                             test_size=test_size, random_state=42, stratify=None)

        # División train/val
        val_size_adjusted = val_size / (1 - test_size)
        X_img_train, X_img_val, X_char_train, X_char_val, y_train, y_val = \
            train_test_split(X_img_temp, X_char_temp, y_temp,
                             test_size=val_size_adjusted, random_state=42)

        print(f"División de datos:")
        print(f"  Entrenamiento: {len(X_img_train):,} muestras")
        print(f"  Validación: {len(X_img_val):,} muestras")
        print(f"  Test: {len(X_img_test):,} muestras")

        return ((X_img_train, X_char_train, y_train),
                (X_img_val, X_char_val, y_val),
                (X_img_test, X_char_test, y_test))

    def train(self,
              train_data: Tuple,
              val_data: Tuple,
              epochs: int = 100,
              batch_size: int = 32,
              callbacks_list: List = None) -> keras.callbacks.History:
        """
        Entrena el modelo con los datos proporcionados.
        """

        X_img_train, X_char_train, y_train = train_data
        X_img_val, X_char_val, y_val = val_data

        if callbacks_list is None:
            callbacks_list = self.get_callbacks()

        # Entrenamiento
        history = self.model.fit(
            x=[X_img_train, X_char_train],
            y=y_train,
            validation_data=([X_img_val, X_char_val], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=1
        )

        return history

    def predict_errors(self,
                       images: np.ndarray,
                       characters: np.ndarray) -> np.ndarray:
        """
        Predice puntuaciones de error para nuevas muestras.

        Returns:
            Array de puntuaciones (N, 10) en escala 0-1
        """

        # Asegurar formato correcto
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=-1)

        predictions = self.model.predict([images, characters])

        return predictions

    def plot_architecture(self):
        """Visualiza la arquitectura del modelo."""
        keras.utils.plot_model(
            self.model,
            to_file='handwriting_error_cnn_architecture.png',
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            dpi=150
        )

    def get_model_summary(self):
        """Imprime resumen detallado del modelo."""
        print("\n" + "=" * 60)
        print("RESUMEN DE ARQUITECTURA CNN PARA ANÁLISIS DE ERRORES")
        print("=" * 60)

        self.model.summary()

        print(f"\nCategorías de error detectadas: {len(ERROR_CATEGORIES)}")
        for i, category in enumerate(ERROR_CATEGORIES):
            print(f"  {i + 1:2d}. {category}")

        print(f"\nDimensiones de entrada:")
        print(f"  - Imagen: {self.image_size} (escala de grises)")
        print(f"  - Carácter: {self.char_vocab_size} (one-hot)")

        print(f"\nSalida: {self.num_error_categories} valores de error (0-1)")


# =============================================================================
# FUNCIONES DE UTILIDAD PARA ANÁLISIS DE RENDIMIENTO
# =============================================================================

def evaluate_model_performance(model: HandwrittingErrorCNN,
                               test_data: Tuple,
                               error_categories: List[str] = None) -> Dict:
    """
    Evalúa el rendimiento del modelo en datos de test.
    """

    if error_categories is None:
        error_categories = ERROR_CATEGORIES

    X_img_test, X_char_test, y_test = test_data

    # Predicciones
    y_pred = model.predict_errors(X_img_test, X_char_test)

    # Métricas por categoría
    results = {}

    for i, category in enumerate(error_categories):
        y_true_cat = y_test[:, i]
        y_pred_cat = y_pred[:, i]

        # Calcular métricas
        mae = np.mean(np.abs(y_true_cat - y_pred_cat))
        mse = np.mean((y_true_cat - y_pred_cat) ** 2)
        rmse = np.sqrt(mse)

        # Correlación
        corr = np.corrcoef(y_true_cat, y_pred_cat)[0, 1]

        results[category] = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'correlation': corr
        }

    return results


def plot_training_history(history: keras.callbacks.History):
    """Visualiza la historia de entrenamiento."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()

    # MAE
    axes[0, 1].plot(history.history['mae'], label='Training MAE')
    axes[0, 1].plot(history.history['val_mae'], label='Validation MAE')
    axes[0, 1].set_title('Mean Absolute Error')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].legend()

    # Learning rate (si está disponible)
    if 'lr' in history.history:
        axes[1, 0].plot(history.history['lr'])
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('LR')
        axes[1, 0].set_yscale('log')

    # Error distribution
    axes[1, 1].hist(history.history['val_loss'], bins=30, alpha=0.7)
    axes[1, 1].set_title('Validation Loss Distribution')
    axes[1, 1].set_xlabel('Loss Value')
    axes[1, 1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()


# =============================================================================
# EJEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    print("Inicializando arquitectura CNN para análisis de errores de escritura...")

    # Crear modelo
    model = HandwrittingErrorCNN(
        image_size=(200, 200),
        num_error_categories=10,
        char_vocab_size=62,
        learning_rate=0.001
    )

    # Mostrar resumen
    model.get_model_summary()

    # Compilar modelo
    model.compile_model()

    print("\n" + "=" * 60)
    print("MODELO LISTO PARA ENTRENAMIENTO")
    print("=" * 60)
    print("\nPróximos pasos:")
    print("1. Cargar datos etiquetados del labeling tool")
    print("2. Preprocesar imágenes a 200x200 píxeles")
    print("3. Codificar caracteres a one-hot")
    print("4. Ejecutar entrenamiento con model.train()")
    print("5. Evaluar rendimiento con evaluate_model_performance()")