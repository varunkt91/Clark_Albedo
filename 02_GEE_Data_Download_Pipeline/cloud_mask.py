#cloud_mask.py

import ee

class Sentinel2CloudMask:
    """
    Attach Sentinel-2 cloud probability images and mask cloudy pixels.
    """

    def __init__(self, clouds, max_cloud_prob):
        """
        Args:
            clouds (ee.ImageCollection): COPERNICUS/S2_CLOUD_PROBABILITY
            max_cloud_prob (int): Cloud probability threshold (0–100)
        """
        self.clouds = clouds
        self.max_cloud_prob = max_cloud_prob

    def attach_cloud_mask(self, img):
        """
        Attach cloud probability image using system:index.
        """
        cloud = self.clouds.filter(
            ee.Filter.equals('system:index', img.get('system:index'))
        ).first()

        return img.set('cloud_mask', cloud)

    def mask_cloudy_pixels(self, img):
        """
        Mask pixels using cloud probability (null-safe).
        """
        cloud = ee.Image(img.get('cloud_mask'))

        # If cloud image is missing, keep image unchanged
        def apply_mask():
            cloud_prob = cloud.select('probability')
            return img.updateMask(cloud_prob.lt(self.max_cloud_prob))

        return ee.Image(
            ee.Algorithms.If(cloud, apply_mask(), img)
        )



