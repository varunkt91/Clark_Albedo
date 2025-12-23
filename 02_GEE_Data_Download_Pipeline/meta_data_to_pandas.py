# meta_data_to_pandas.py
import ee

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
