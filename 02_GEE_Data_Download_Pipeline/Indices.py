# def NDVI_img(image):
  
#   ndvi = image.normalizedDifference(['B8', 'B4']).select([0], ['ndvi']).set('time_start',image.get('system:time_start'))
#   return image.addBands(ndvi)

# # Enhanced Vegetation Index
# def EVI_img(image):
#     # Compute the EVI using an expression.
#     EVI = image.expression(
#         '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
#             'NIR': image.select('B8'),
#             'RED': image.select('B4'),
#             'BLUE': image.select('B2')
#         }).rename("evi")

#     return image.addBands(EVI)

# # Function to compute hillshade from SRTM
# def add_hillshade(img):
#     """Adds a hillshade band from SRTM DEM to the given image."""
#     srtm = ee.Image("USGS/SRTMGL1_003")
#     hillshade = ee.Terrain.hillshade(srtm).rename("hillshade")
#     return img.addBands(hillshade)


# # Normalized Difference Snow Index (NDSI) for Sentinel-2
# def NDSI_img(image):
#     # Compute NDSI
#     NDSI = image.normalizedDifference(['B3', 'B11']).rename("ndsi")
#     return image.addBands(NDSI)

# #Land Surface Water Index
# def LSWI_img(image):
 
#   lswi = image.normalizedDifference(['B8', 'B11']).select([0], ['lswi']).set('time_start',image.get('system:time_start')).rename("lswi")
#   return image.addBands(lswi)

# # #Simple Ratio Water Index
# def SRWI_img(image):
#   SRWI = image.expression(
#         '(NIR1/NIR2)', {
#             'NIR1': image.select('B8'),
#             'NIR2': image.select('sur_refl_b05')
            
#         }).rename("srwi")
#   return image.addBands(SRWI)

# # Simple Ratio Tillage Index
# def SRTI_img(image):
#   SRTI = image.expression(
#         '(SWIR1/SWIR2)', {
#             'SWIR1': image.select('B11'),
#             'SWIR2': image.select('B12')
            
#         }).rename("srti")
#   return image.addBands(SRTI)

# #Normalized difference Tillage Index
# def NDTI_img(image):
 
#   ndti = image.normalizedDifference(['B11', 'B12']).select([0], ['ndti']).set('time_start',image.get('system:time_start')).rename("ndti")
#   return image.addBands(ndti)


# # Crop residue cover index
# def CRCI_img(image):

#   crci = image.normalizedDifference(['B11', 'B2']).select([0], ['crci']).set('time_start',image.get('system:time_start')).rename("crci")
#   return image.addBands(crci)

# # Modified CRC index
# def MCRC_img(image):
  
#   mcrc = image.normalizedDifference(['B11', 'B3']).select([0], ['mcrc']).set('time_start',image.get('system:time_start')).rename("mcrc")
#   return image.addBands(mcrc)

# #Soil Adjusted Vegetation Index
# def SAVI_img(image):
#     # Compute the EVI using an expression.
#     SAVI = image.expression(
#         '((NIR-RED)/(NIR+RED+0.5))*1.5', {
#             'NIR': image.select('B8'),
#             'RED': image.select('B4')
#         }).rename("savi")
#     return image.addBands(SAVI)

# # Normalized Difference Senescent VegetationIndex
# # 
# def NDSVI_img(image):
#   NDSVI = image.normalizedDifference(['B11', 'B4']).select([0], ['ndsvi']).set('time_start',image.get('system:time_start')).rename("ndsvi")
#   return image.addBands(NDSVI)


import ee

class Sentinel2Indices:
    """Class to compute Sentinel-2 indices."""

    @staticmethod
    def NDVI(img):
        ndvi = img.normalizedDifference(['B8','B4']).rename('ndvi')
        return img.addBands(ndvi)

    @staticmethod
    def EVI(img):
        evi = img.expression(
            '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
            {'NIR': img.select('B8'),
             'RED': img.select('B4'),
             'BLUE': img.select('B2')}
        ).rename('evi')
        return img.addBands(evi)

    @staticmethod
    def LSWI(img):
        lswi = img.normalizedDifference(['B8','B11']).rename('lswi')
        return img.addBands(lswi)

    @staticmethod
    def SRTI(img):
        srti = img.normalizedDifference(['B11','B12']).rename('srti')
        return img.addBands(srti)
    
    @staticmethod
    def SRWI(img):
        srwi = img.normalizedDifference(['B8','B5']).rename('srwi')
        return img.addBands(srwi)

    @staticmethod
    def NDTI(img):
        ndti = img.normalizedDifference(['B11','B12']).rename('ndti')
        return img.addBands(ndti)

    @staticmethod
    def CRCI(img):
        crci = img.normalizedDifference(['B11','B2']).rename('crci')
        return img.addBands(crci)

    @staticmethod
    def MCRC(img):
        mcrc = img.normalizedDifference(['B11','B3']).rename('mcrc')
        return img.addBands(mcrc)

    @staticmethod
    def SAVI(img):
        savi = img.expression(
            '((NIR-RED)/(NIR+RED+0.5))*1.5',
            {'NIR': img.select('B8'),
             'RED': img.select('B4')}
        ).rename('savi')
        return img.addBands(savi)

    @staticmethod
    def NDSVI(img):
        ndsvi = img.normalizedDifference(['B11','B4']).rename('ndsvi')
        return img.addBands(ndsvi)

    @staticmethod
    def NDSI(img):
        ndsi = img.normalizedDifference(['B3','B11']).rename('ndsi')
        return img.addBands(ndsi)

    @staticmethod
    def add_hillshade(img):
        srtm = ee.Image("USGS/SRTMGL1_003")
        hillshade = ee.Terrain.hillshade(srtm).rename("hillshade")
        return img.addBands(hillshade)

