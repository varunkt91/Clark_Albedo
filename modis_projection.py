# function to conver S2 to modis projection

def reproject_s2_to_modis_10m(s2_image, modis_proj):
    """
    Reprojects a Sentinel-2 image to the MODIS CRS at 10m resolution.
    
    Parameters:
    - s2_image: ee.Image, the Sentinel-2 image to reproject
    - modis_proj: ee.Projection, projection object from MODIS data

    Returns:
    - ee.Image: Reprojected image with 10m scale
    """
    return s2_image.resample('bilinear').reproject(crs=modis_proj.crs(), scale=10)