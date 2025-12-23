# metadata_manager.py

import ee
import pandas as pd


class MetadataExtractor:
    """
    Extracts metadata from an Earth Engine ImageCollection.
    """

    @staticmethod
    def extract_metadata(img):
        """
        Extract image ID and acquisition date.
        """
        # Try to get system:time_start
        time_start = img.get('system:time_start')

        # If it exists, convert to human-readable date
        date = ee.Date(time_start).format('YYYY-MM-dd') if time_start else ee.String(img.id())

        return ee.Feature(None, {
            'id': img.id(),
            'system_time_start': time_start,
            'date': date
        })


class GEECollectionMetadataManager:
    """
    Extracts image-level metadata from an ee.ImageCollection,
    converts it to a Pandas DataFrame, and optionally saves to CSV.
    """

    def __init__(self, collection, extractor=MetadataExtractor.extract_metadata, csv_path=None):
        """
        Parameters
        ----------
        collection : ee.ImageCollection
            Image collection to extract metadata from
        extractor : callable
            Function that takes ee.Image and returns ee.Feature
        csv_path : str, optional
            Path to save CSV output
        """
        self.collection = collection
        self.extractor = extractor
        self.csv_path = csv_path

    def to_feature_collection(self):
        """Convert ImageCollection to FeatureCollection"""
        return ee.FeatureCollection(
            self.collection.map(self.extractor)
        )

    def to_dataframe(self):
        """Extract metadata to a Pandas DataFrame"""
        fc = self.to_feature_collection()
        metadata_list = fc.getInfo()["features"]

        df = pd.DataFrame([f["properties"] for f in metadata_list])

        # Convert timestamps if present
        if "system_time_start" in df.columns:
            df["system_time_start"] = pd.to_datetime(
                df["system_time_start"],
                unit="ms",
                errors="coerce"
            )

        return df

    def save_csv(self, df):
        """Save DataFrame to CSV if path is provided"""
        if self.csv_path:
            df.to_csv(self.csv_path, index=False)

    def run(self, preview=True):
        """
        Full pipeline:
        ImageCollection → FeatureCollection → DataFrame → CSV
        """
        df = self.to_dataframe()
        self.save_csv(df)

        if preview:
            print(df.head())

        return df
