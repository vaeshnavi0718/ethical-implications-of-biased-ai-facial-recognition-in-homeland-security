import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import face_recognition
from typing import List, Tuple, Dict, Optional

class DataProcessor:
    def __init__(self, data_dir: str):
        """
        Initialize the DataProcessor class.
        
        Args:
            data_dir (str): Path to the data directory
        """
        self.data_dir = Path(data_dir)
        self.face_encodings = {}
        self.metadata = {}
        
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess an image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            np.ndarray: Preprocessed image
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            List[Tuple[int, int, int, int]]: List of face locations (top, right, bottom, left)
        """
        return face_recognition.face_locations(image)
    
    def extract_face_encoding(self, image: np.ndarray, face_location: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract face encoding from a detected face.
        
        Args:
            image (np.ndarray): Input image
            face_location (Tuple[int, int, int, int]): Face location coordinates
            
        Returns:
            np.ndarray: Face encoding
        """
        return face_recognition.face_encodings(image, [face_location])[0]
    
    def process_dataset(self, metadata_file: Optional[str] = None) -> pd.DataFrame:
        """
        Process the entire dataset and extract face encodings.
        
        Args:
            metadata_file (Optional[str]): Path to metadata file containing demographic information
            
        Returns:
            pd.DataFrame: Processed dataset with face encodings and metadata
        """
        processed_data = []
        
        # Load metadata if provided
        if metadata_file:
            self.metadata = pd.read_csv(metadata_file)
        
        # Process each image in the data directory
        for image_path in self.data_dir.glob('**/*.jpg'):
            try:
                image = self.load_image(str(image_path))
                face_locations = self.detect_faces(image)
                
                if not face_locations:
                    continue
                    
                # Use the first detected face
                face_location = face_locations[0]
                face_encoding = self.extract_face_encoding(image, face_location)
                
                # Get metadata for this image if available
                image_metadata = {}
                if metadata_file:
                    image_id = image_path.stem
                    image_metadata = self.metadata[self.metadata['image_id'] == image_id].to_dict('records')[0]
                
                processed_data.append({
                    'image_path': str(image_path),
                    'face_encoding': face_encoding,
                    **image_metadata
                })
                
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                continue
        
        return pd.DataFrame(processed_data)
    
    def save_processed_data(self, output_file: str):
        """
        Save processed data to a file.
        
        Args:
            output_file (str): Path to save the processed data
        """
        if not hasattr(self, 'processed_data'):
            raise ValueError("No processed data available. Run process_dataset first.")
            
        # Convert face encodings to list for storage
        data_to_save = self.processed_data.copy()
        data_to_save['face_encoding'] = data_to_save['face_encoding'].apply(lambda x: x.tolist())
        
        data_to_save.to_pickle(output_file)
    
    def load_processed_data(self, input_file: str) -> pd.DataFrame:
        """
        Load processed data from a file.
        
        Args:
            input_file (str): Path to the processed data file
            
        Returns:
            pd.DataFrame: Loaded processed data
        """
        self.processed_data = pd.read_pickle(input_file)
        # Convert face encodings back to numpy arrays
        self.processed_data['face_encoding'] = self.processed_data['face_encoding'].apply(np.array)
        return self.processed_data 