import ee
from cloud_mask import Sentinel2CloudMask  # Your cloud mask class

class RasterCollection:
    """Class to load and process raster collections (Sentinel-2, MODIS) with indices and cloud masking."""

    def __init__(self, config, indices_class=None):
        """
        Args:
            config: module containing user configuration (dates, region, cloud thresholds, clip func, etc.)
            indices_class: class containing Sentinel-2 indices functions
        """
        self.start_date = config.startDate
        self.end_date = config.endDate
        self.region = config.region
        self.clip_func = config.clip_to_roi
        self.cloud_cover = config.cloud_cover
        self.cloud_max_prob = config.max_cloud_probability
        self.indices_class = indices_class

        self.s2_collection = None
        self.clouds_collection = None
        self.cloud_masker = None
        self.modis_collection = None

    # -----------------------------
    # Sentinel-2 Methods
    # -----------------------------
    def load_sentinel2(self):
        """Load Sentinel-2 SR collection with cloud cover filtering and apply indices."""
        ic = (
            ee.ImageCollection("COPERNICUS/S2_SR")
            .filterDate(self.start_date, self.end_date)
            .filterBounds(self.region)
            .map(lambda img: self.clip_func(img, self.region))
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', self.cloud_cover))
        )

        # Apply all indices from the indices class dynamically
        if self.indices_class:
            for func_name in dir(self.indices_class):
                if not func_name.startswith("__") and callable(getattr(self.indices_class, func_name)):
                    ic = ic.map(getattr(self.indices_class, func_name))

        self.s2_collection = ic
        return self.s2_collection

    def load_clouds(self):
        """Load Sentinel-2 cloud probability collection and initialize cloud masker."""
        self.clouds_collection = (
            ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
            .filterDate(self.start_date, self.end_date)
            .filterBounds(self.region)
        )

        self.cloud_masker = Sentinel2CloudMask(
            clouds=self.clouds_collection,
            max_cloud_prob=self.cloud_max_prob
        )
        return self.clouds_collection

    def attach_cloud_mask(self, img):
        """Attach cloud probability image to each Sentinel-2 image."""
        cloud = self.clouds_collection.filter(
            ee.Filter.equals('system:index', img.get('system:index'))
        ).first()
        return img.set('cloud_mask', cloud)

    def mask_clouds(self, img):
        """Apply cloud masking using attached cloud probability image."""
        return self.cloud_masker.mask_cloudy_pixels(img)

    def process_sentinel2(self):
        """Full Sentinel-2 pipeline: load, indices, attach cloud mask, apply masking."""
        if self.s2_collection is None:
            self.load_sentinel2()
        if self.cloud_masker is None:
            self.load_clouds()

        s2_with_clouds = self.s2_collection.map(self.attach_cloud_mask)
        s2_masked = s2_with_clouds.map(self.mask_clouds)
        return s2_masked

    # -----------------------------
    # MODIS Methods
    # -----------------------------
    def load_modis(self, band_name='Albedo_WSA_shortwave', format_date_func=None):
        """Load MODIS collection, clip to region, and optionally format dates."""
        self.modis_collection = (
            ee.ImageCollection('MODIS/061/MCD43A3')
            .select(band_name)
            .filterDate(self.start_date, self.end_date)
            .filterBounds(self.region)
            .map(lambda img: self.clip_func(img, self.region))
        )
        if format_date_func:
            self.modis_collection = self.modis_collection.map(format_date_func)
        return self.modis_collection
