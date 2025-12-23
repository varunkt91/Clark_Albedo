#Indices.py


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

