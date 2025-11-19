# dataset_processor.py - Processeur de dataset CSV pour images PGPR

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import shutil

class CSVDatasetProcessor:
    """Processeur de dataset CSV pour les images PGPR"""
    
    def __init__(self, images_dir: str, train_csv: str, test_csv: str, validation_csv: str = None):
        self.images_dir = Path(images_dir)
        self.train_csv = Path(train_csv)
        self.test_csv = Path(test_csv)
        self.validation_csv = Path(validation_csv) if validation_csv else None
        
        # Charger les CSV
        self.train_df = pd.read_csv(self.train_csv)
        self.test_df = pd.read_csv(self.test_csv)
        self.validation_df = pd.read_csv(self.validation_csv) if self.validation_csv else None
        
        self.bacteria_types = [col for col in self.train_df.columns if col != 'filename']
        
        print(f"CSVDatasetProcessor initialisé. Types de bactéries: {self.bacteria_types}")
        self._validate_dataset()
    
    def _validate_dataset(self):
        """Valide la présence des images et la cohérence des CSV"""
        print("Validation du dataset...")
        
        all_filenames = set(self.train_df['filename'].tolist())
        if self.test_df is not None:
            all_filenames.update(self.test_df['filename'].tolist())
        if self.validation_df is not None:
            all_filenames.update(self.validation_df['filename'].tolist())
        
        missing_images = []
        for filename in all_filenames:
            if not (self.images_dir / filename).exists():
                missing_images.append(filename)
        
        if missing_images:
            print(f"⚠️ {len(missing_images)} images référencées dans les CSV sont manquantes dans le dossier {self.images_dir}. Exemples: {missing_images[:5]}")
        else:
            print("✅ Toutes les images référencées sont présentes.")
    
    def get_bacteria_labels(self, filename: str, split: str) -> Optional[Dict[str, int]]:
        """Récupère les labels de bactéries pour un fichier donné et un split"""
        df = None
        if split == "train":
            df = self.train_df
        elif split == "test":
            df = self.test_df
        elif split == "validation":
            df = self.validation_df
        
        if df is not None:
            row = df[df['filename'] == filename]
            if not row.empty:
                labels = row[self.bacteria_types].iloc[0].to_dict()
                return labels
        return None
    
    def get_all_image_paths_with_labels(self, split: Optional[str] = None) -> List[Tuple[Path, Dict[str, int], str]]:
        """Retourne tous les chemins d'images avec leurs labels et leur split"""
        data = []
        
        splits_to_process = []
        if split:
            splits_to_process.append(split)
        else:
            splits_to_process.extend(["train", "test"])
            if self.validation_df is not None:
                splits_to_process.append("validation")
        
        for current_split in splits_to_process:
            df = None
            if current_split == "train":
                df = self.train_df
            elif current_split == "test":
                df = self.test_df
            elif current_split == "validation":
                df = self.validation_df
            
            if df is not None:
                for _, row in df.iterrows():
                    filename = row['filename']
                    image_path = self.images_dir / filename
                    labels = row[self.bacteria_types].to_dict()
                    data.append((image_path, labels, current_split))
        return data
    
    def get_dataset_stats(self) -> Dict:
        """Retourne des statistiques détaillées sur le dataset"""
        stats = {}
        
        for split_name, df in [("train", self.train_df), ("test", self.test_df), ("validation", self.validation_df)]:
            if df is not None:
                total_images = len(df)
                bacteria_counts = {b: df[b].sum() for b in self.bacteria_types}
                
                stats[split_name] = {
                    "total_images": total_images,
                    "bacteria_counts": bacteria_counts
                }
        return stats

def analyze_csv_dataset(train_csv: str, test_csv: str, validation_csv: str = None):
    """Fonction utilitaire pour analyser un dataset CSV"""
    print("Analyse du dataset CSV...")
    
    try:
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
        validation_df = pd.read_csv(validation_csv) if validation_csv else None
        
        bacteria_types = [col for col in train_df.columns if col != 'filename']
        print(f"Types de bactéries détectés: {bacteria_types}")
        
        print("\nStatistiques par split:")
        for name, df in [("Train", train_df), ("Test", test_df)]:
            if df is not None:
                print(f"  {name}: {len(df)} images")
                for b_type in bacteria_types:
                    count = df[b_type].sum()
                    print(f"    - {b_type}: {count} ({count/len(df)*100:.1f}%)")
        
        if validation_df is not None:
            print(f"  Validation: {len(validation_df)} images")
            for b_type in bacteria_types:
                count = validation_df[b_type].sum()
                print(f"    - {b_type}: {count} ({count/len(validation_df)*100:.1f}%)")
        
        print("\nAnalyse terminée.")
    except Exception as e:
        print(f"Erreur lors de l'analyse du dataset CSV: {e}")

if __name__ == "__main__":
    # Exemple d'utilisation (à adapter)
    IMAGES_DIR = "./pgpr_images/images/"
    TRAIN_CSV = "./pgpr_images/train_labels.csv"
    TEST_CSV = "./pgpr_images/test_labels.csv"
    
    if Path(TRAIN_CSV).exists() and Path(TEST_CSV).exists() and Path(IMAGES_DIR).exists():
        processor = CSVDatasetProcessor(IMAGES_DIR, TRAIN_CSV, TEST_CSV)
        stats = processor.get_dataset_stats()
        print("\nStatistiques du dataset:")
        for split, data in stats.items():
            print(f"  {split.title()}:")
            print(f"    Total images: {data['total_images']}")
            print(f"    Bacteria counts: {data['bacteria_counts']}")
        
        # Exemple de récupération de labels
        sample_filename = processor.train_df['filename'].iloc[0]
        labels = processor.get_bacteria_labels(sample_filename, "train")
        print(f"\nLabels pour {sample_filename}: {labels}")
    else:
        print("Veuillez configurer les chemins IMAGES_DIR, TRAIN_CSV, TEST_CSV pour l'exemple.")
