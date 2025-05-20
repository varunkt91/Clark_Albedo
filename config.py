# config.py

import ee
import geemap

# Map setup (optional default settings)
Map = geemap.Map(center=[-124.3998113967669, 46.328967879608506], zoom=8)


# Initialize Earth Engine (ensure this is done somewhere in your main script too)
ee.Initialize()  # Optional here


startDate = '2021-01-01'
endDate = '2021-06-30'
cloud_cover = 5
# Define region of interest (example: a rectangle)
region = ee.Geometry.Polygon(
    [[
        [-124.3998113967669, 46.328967879608506],
        [-124.3998113967669, 41.620634777502005],
        [-116.4457098342669, 41.620634777502005],
        [-116.4457098342669, 46.328967879608506],
        [-124.3998113967669, 46.328967879608506]  # Closing the loop
    ]]
)

# clip image within roi

def clip_to_roi(image, roi):
    return image.clip(roi)