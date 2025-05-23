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



def mask_cloudy_pixels(img, max_cloud_prob=20):
    """
    Masks out pixels in a Sentinel-2 image where cloud probability exceeds the threshold.
    
    Args:
        img (ee.Image): A Sentinel-2 image with a 'cloud_mask' image attached as a property.
        max_cloud_prob (int): Cloud probability threshold (0–100).
    
    Returns:
        ee.Image: Cloud-masked Sentinel-2 image.
    """
    cloud_prob = ee.Image(img.get('cloud_mask')).select('probability')
    mask = cloud_prob.lt(max_cloud_prob)
    return img.updateMask(mask)


