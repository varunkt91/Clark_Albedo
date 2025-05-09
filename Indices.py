def NDVI_img(image):
  
  ndvi = image.normalizedDifference(['B8', 'B4']).select([0], ['ndvi']).set('time_start',image.get('system:time_start'))
  return image.addBands(ndvi)

# Enhanced Vegetation Index
def EVI_img(image):
    # Compute the EVI using an expression.
    EVI = image.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
            'NIR': image.select('B8'),
            'RED': image.select('B4'),
            'BLUE': image.select('B2')
        }).rename("evi")

    return image.addBands(EVI)

#Land Surface Water Index
def LSWI_img(image):
 
  lswi = image.normalizedDifference(['B8', 'B11']).select([0], ['lswi']).set('time_start',image.get('system:time_start')).rename("lswi")
  return image.addBands(lswi)

# #Simple Ratio Water Index
def SRWI_img(image):
  SRWI = image.expression(
        '(NIR1/NIR2)', {
            'NIR1': image.select('B8'),
            'NIR2': image.select('sur_refl_b05')
            
        }).rename("srwi")
  return image.addBands(SRWI)

# Simple Ratio Tillage Index
def SRTI_img(image):
  SRTI = image.expression(
        '(SWIR1/SWIR2)', {
            'SWIR1': image.select('B11'),
            'SWIR2': image.select('B12')
            
        }).rename("srti")
  return image.addBands(SRTI)

#Normalized difference Tillage Index
def NDTI_img(image):
 
  ndti = image.normalizedDifference(['B11', 'B12']).select([0], ['ndti']).set('time_start',image.get('system:time_start')).rename("ndti")
  return image.addBands(ndti)


# Crop residue cover index
def CRCI_img(image):

  crci = image.normalizedDifference(['B11', 'B2']).select([0], ['crci']).set('time_start',image.get('system:time_start')).rename("crci")
  return image.addBands(crci)

# Modified CRC index
def MCRC_img(image):
  
  mcrc = image.normalizedDifference(['B11', 'B3']).select([0], ['mcrc']).set('time_start',image.get('system:time_start')).rename("mcrc")
  return image.addBands(mcrc)

#Soil Adjusted Vegetation Index
def SAVI_img(image):
    # Compute the EVI using an expression.
    SAVI = image.expression(
        '((NIR-RED)/(NIR+RED+0.5))*1.5', {
            'NIR': image.select('B8'),
            'RED': image.select('B4')
        }).rename("savi")
    return image.addBands(SAVI)

# Normalized Difference Senescent VegetationIndex
# 
def NDSVI_img(image):
  NDSVI = image.normalizedDifference(['B11', 'B4']).select([0], ['ndsvi']).set('time_start',image.get('system:time_start')).rename("ndsvi")
  return image.addBands(NDSVI)