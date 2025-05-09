import ee
def maskS2clouds(image):
    scl = image.select('SCL')

    # Define mask to exclude unwanted classes
    mask = scl.neq(3).And(
           scl.neq(8)).And(
           scl.neq(9)).And(
           scl.neq(10)).And(
           scl.neq(11)).And(
           scl.neq(1))

    return (image.updateMask(mask)
                 .select('B.*')
                 .copyProperties(image, ['system:time_start']))



# Apply quality mask: keep only BRDF_Albedo_Band_Quality == 0
def mask_modis_clouds(pair):
    albedo = ee.Image(pair.get('primary'))
    qa = ee.Image(pair.get('secondary'))
    mask = qa.eq(0)  # 0 = good quality
    return albedo.updateMask(mask).copyProperties(albedo, ['system:time_start'])

