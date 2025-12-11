import ee
# Extract metadata for each image
def extract_metadata(img):
    return ee.Feature(None, {
        'id': img.id(),
        'system_time_start': img.get('system:time_start'),
        'date': img.get('date')
    })