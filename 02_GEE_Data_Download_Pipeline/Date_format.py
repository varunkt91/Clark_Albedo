import ee

def format_date(img):
    date = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd')
    return img.set('date', date)