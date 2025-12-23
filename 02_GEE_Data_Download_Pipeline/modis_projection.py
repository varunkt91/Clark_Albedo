# # function to convert S2 to modis projection

# import ee

# # Function to reproject Sentinel-2 to MODIS CRS (define this elsewhere)
# def reproject_s2_to_modis_500m(s2_img, modis_img):
#     modis_proj = modis_img.projection()
#     modis_transform = modis_proj.getInfo()['transform']
#     s2_reprojected = s2_img.resample('bilinear').reproject(
#         crs=modis_proj.crs(),
#         crsTransform=modis_transform,
        
#     )
#     return s2_reprojected

# # Function to reproject Sentinel-2 to WGS84


# def reproject_s2_to_wgs84(s2_img, scale_or_image=500):
#     """
#     Reprojects a Sentinel-2 image to WGS84 (EPSG:4326).

#     Args:
#         s2_img: ee.Image, Sentinel-2 image
#         scale_or_image: float (pixel size in meters) OR ee.Image (to match its resolution)

#     Returns:
#         ee.Image reprojected to WGS84
#     """
#     # Determine numeric scale
#     if isinstance(scale_or_image, ee.Image):
#         # Get the nominal scale of the image (meters)
#         scale = scale_or_image.projection().nominalScale()
#     else:
#         # Use numeric scale
#         scale = float(scale_or_image)
    
#     # Reproject
#     s2_reprojected = s2_img.resample('bilinear').reproject(
#         crs='EPSG:4326',
#         scale=scale
#     )
#     return s2_reprojected


# def reproject_s2_to_modis_scale_10m(s2_img, modis_img):
#     # Get MODIS CRS
#     modis_proj = modis_img.projection()
#     modis_crs = modis_proj.crs()

#     # Get S2 native scale (ee.Number)
#     s2_proj = s2_img.select(0).projection()
#     s2_scale = s2_proj.nominalScale()

#     # Get MODIS transform info (client-side)
#     modis_info = modis_proj.getInfo()
#     modis_transform = modis_info['transform']
#     translate_x = modis_transform[2]
#     translate_y = modis_transform[5]

#     # Convert s2_scale to float
#     s2_scale_val = s2_scale.getInfo()

#     # Construct new transform using S2 scale and MODIS origin
#     new_transform = [
#         s2_scale_val, 0, translate_x,
#         0, -s2_scale_val, translate_y
#     ]
    
#     # Reproject to MODIS CRS with Sentinel-2 resolution
#     s2_reprojected = s2_img.resample('bilinear').reproject(
#         crs=modis_crs,
#         crsTransform=new_transform
#     )

#     return s2_reprojected

import ee

class ModisReprojector:
    """
    Utility class for reprojection between Sentinel-2 and MODIS grids.
    """

    # -------------------------------------------------
    # S2 → MODIS (500 m)
    # -------------------------------------------------
    @staticmethod
    def s2_to_modis_500m(s2_img, modis_img):
        """
        Reproject Sentinel-2 image to MODIS CRS and 500 m grid.
        """
        modis_proj = modis_img.projection()
        modis_transform = modis_proj.getInfo()['transform']

        return (
            s2_img
            .resample('bilinear')
            .reproject(
                crs=modis_proj.crs(),
                crsTransform=modis_transform
            )
        )

    # -------------------------------------------------
    # S2 → WGS84 (custom scale)
    # -------------------------------------------------
    @staticmethod
    def s2_to_wgs84(s2_img, scale_or_image=500):
        """
        Reproject Sentinel-2 to WGS84 (EPSG:4326).

        scale_or_image:
            - float → output pixel size (meters)
            - ee.Image → match its nominal scale
        """
        if isinstance(scale_or_image, ee.Image):
            scale = scale_or_image.projection().nominalScale()
        else:
            scale = float(scale_or_image)

        return (
            s2_img
            .resample('bilinear')
            .reproject(
                crs='EPSG:4326',
                scale=scale
            )
        )

    # -------------------------------------------------
    # S2 → MODIS CRS, Sentinel-2 resolution (10 m)
    # -------------------------------------------------
    @staticmethod
    def s2_to_modis_10m(s2_img, modis_img):
        """
        Reproject Sentinel-2 into MODIS CRS
        while keeping Sentinel-2 native resolution.
        """
        modis_proj = modis_img.projection()
        modis_crs = modis_proj.crs()

        # Sentinel-2 scale
        s2_scale = s2_img.select(0).projection().nominalScale()
        s2_scale_val = s2_scale.getInfo()

        # MODIS transform (origin)
        modis_transform = modis_proj.getInfo()['transform']
        translate_x = modis_transform[2]
        translate_y = modis_transform[5]

        # New transform: S2 resolution + MODIS origin
        new_transform = [
            s2_scale_val, 0, translate_x,
            0, -s2_scale_val, translate_y
        ]

        return (
            s2_img
            .resample('bilinear')
            .reproject(
                crs=modis_crs,
                crsTransform=new_transform
            )
        )
