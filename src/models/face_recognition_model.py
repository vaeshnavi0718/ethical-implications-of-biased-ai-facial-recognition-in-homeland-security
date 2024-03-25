import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict, Optional
import tensorflow as tf
from tensorflow.keras import layers, Model
import pandas as pd

class BiasAwareFaceRecognitionModel:
    def __init__(self, embedding_dim: int = 128):
        """
        Initialize the BiasAwareFaceRecognitionModel.
        
        Args:
            embedding_dim (int): Dimension of face embeddings
        """
        self.embedding_dim = embedding_dim
        self.model = self._build_model()
        self.face_database = {}
        self.sensitive_attributes = {}
        
    def _build_model(self) -> Model:
        """
        Build the face recognition model architecture.
        
        Returns:
            Model: Compiled Keras model
        """
        # Input layer
        input_layer = layers.Input(shape=(128,))  # Face recognition default encoding size
        
        # Dense layers for feature extraction
        x = layers.Dense(256, activation='relu')(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Output embedding layer
        output_layer = layers.Dense(self.embedding_dim, activation=None)(x)
        
        # Create and compile model
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            optimizer='adam',
            loss='cosine_similarity'
        )
        
        return model
    
    def train(self, 
             face_encodings: np.ndarray,
             labels: np.ndarray,
             sensitive_attributes: Optional[Dict[str, np.ndarray]] = None,
             batch_size: int = 32,
             epochs: int = 10):
        """
        Train the face recognition model with bias awareness.
        
        Args:
            face_encodings (np.ndarray): Face encodings for training
            labels (np.ndarray): Identity labels
            sensitive_attributes (Optional[Dict[str, np.ndarray]]): Dictionary of sensitive attributes
            batch_size (int): Batch size for training
            epochs (int): Number of training epochs
        """
        # Store sensitive attributes if provided
        if sensitive_attributes:
            self.sensitive_attributes = sensitive_attributes
        
        # Convert labels to one-hot encoding
        num_classes = len(np.unique(labels))
        y_one_hot = tf.keras.utils.to_categorical(labels, num_classes)
        
        # Train the model
        self.model.fit(
            face_encodings,
            y_one_hot,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2
        )
    
    def add_to_database(self, 
                       face_encoding: np.ndarray,
                       identity: str,
                       sensitive_attributes: Optional[Dict[str, any]] = None):
        """
        Add a face encoding to the database.
        
        Args:
            face_encoding (np.ndarray): Face encoding to add
            identity (str): Identity label
            sensitive_attributes (Optional[Dict[str, any]]): Sensitive attributes for this face
        """
        self.face_database[identity] = face_encoding
        if sensitive_attributes:
            self.sensitive_attributes[identity] = sensitive_attributes
    
    def recognize_face(self, 
                      face_encoding: np.ndarray,
                      threshold: float = 0.6) -> Tuple[Optional[str], float]:
        """
        Recognize a face from the database.
        
        Args:
            face_encoding (np.ndarray): Face encoding to recognize
            threshold (float): Similarity threshold for recognition
            
        Returns:
            Tuple[Optional[str], float]: (Recognized identity, confidence score)
        """
        if not self.face_database:
            raise ValueError("Face database is empty")
        
        # Get embeddings for the input face
        input_embedding = self.model.predict(np.expand_dims(face_encoding, axis=0))[0]
        
        # Calculate similarities with all faces in database
        similarities = {}
        for identity, db_encoding in self.face_database.items():
            db_embedding = self.model.predict(np.expand_dims(db_encoding, axis=0))[0]
            similarity = cosine_similarity([input_embedding], [db_embedding])[0][0]
            similarities[identity] = similarity
        
        # Find the best match
        best_match = max(similarities.items(), key=lambda x: x[1])
        
        if best_match[1] >= threshold:
            return best_match
        return None, 0.0
    
    def analyze_demographic_performance(self) -> pd.DataFrame:
        """
        Analyze model performance across different demographic groups.
        
        Returns:
            pd.DataFrame: Performance metrics by demographic group
        """
        if not self.sensitive_attributes:
            raise ValueError("No sensitive attributes available for analysis")
        
        results = []
        for attr_name, attr_values in self.sensitive_attributes.items():
            unique_values = np.unique(attr_values)
            for value in unique_values:
                # Get faces for this demographic group
                group_faces = [face for face, attrs in self.sensitive_attributes.items()
                             if attrs[attr_name] == value]
                
                # Calculate performance metrics for this group
                # (This is a placeholder - implement actual metrics calculation)
                results.append({
                    'attribute': attr_name,
                    'value': value,
                    'count': len(group_faces),
                    'recognition_rate': 0.0,  # Implement actual calculation
                    'false_positive_rate': 0.0,  # Implement actual calculation
                    'false_negative_rate': 0.0  # Implement actual calculation
                })
        
        return pd.DataFrame(results)
    
    def save_model(self, filepath: str):
        """
        Save the model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        self.model.save(filepath)
    
    def load_model(self, filepath: str):
        """
        Load the model from disk.
        
        Args:
            filepath (str): Path to load the model from
        """
        self.model = tf.keras.models.load_model(filepath) 