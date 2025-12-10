import ee

# Merge matched pairs
# def merge_images(pair):
#     s2_img = ee.Image(pair.get('primary'))
#     modis_img = ee.Image(pair.get('secondary'))
    
#     modis_proj = modis_img.projection()
    
#     s2_resampled = s2_img.resample('bilinear').reproject(crs=modis_proj.crs(), scale=10)
    
#     # Create combined mask
#     overlap_mask = s2_resampled.select('B4').mask().And(modis_img.mask())
    
#     s2_masked = s2_resampled.updateMask(overlap_mask)
#     modis_masked = modis_img.updateMask(overlap_mask).rename('MODIS_Albedo_WSA_shortwave')
    
#     return (s2_masked.addBands(modis_masked)
#                      .copyProperties(s2_img, ['system:time_start', 'date']))






# def merge_images(pair):
#     s2_img = ee.Image(pair.get('primary'))
#     modis_img = ee.Image(pair.get('secondary'))

#     modis_proj = modis_img.projection()

#     # Reproject S2 to match MODIS projection (includes CRS and nominal scale)
#     s2_resampled = s2_img.resample('bilinear').reproject(modis_proj)

#     # Combined valid mask from both images
#     overlap_mask = s2_resampled.select('B4').mask().And(modis_img.mask())

#     s2_masked = s2_resampled.updateMask(overlap_mask)
#     modis_masked = modis_img.updateMask(overlap_mask).rename('MODIS_Albedo_WSA_shortwave')

#     return (s2_masked.addBands(modis_masked)
#                      .copyProperties(s2_img, ['system:time_start', 'date']))


def merge_images(pair):
    # Get Sentinel-2 and MODIS images from the pair
    s2_img = ee.Image(pair.get('primary'))
    modis_img = ee.Image(pair.get('secondary'))

    # Create a mask of common valid pixels in both images (no reprojection)
    common_mask = s2_img.select('B4').mask().And(modis_img.mask())

    # Apply the common mask to both images
    s2_masked = s2_img.updateMask(common_mask)
    modis_masked = modis_img.updateMask(common_mask).rename('MODIS_Albedo_WSA_shortwave')

    # Merge the masked bands and retain relevant metadata
    return (s2_masked.addBands(modis_masked)
                   .copyProperties(s2_img, ['system:time_start', 'date']))


