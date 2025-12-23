class VisualizationParams:
    """
    Centralized visualization parameters for Earth Engine layers
    """

    @staticmethod
    def modis_albedo_raw():
        """MODIS albedo (raw values, no rescaling)"""
        return {
            'min': 0,
            'max': 400,   # raw MODIS range (~0–0.4)
            'palette': [
                '0d0887',
                '6a00a8',
                'b12a90',
                'e16462',
                'fca636',
                'f0f921'
            ]
        }

    @staticmethod
    def sentinel2_rgb():
        """Sentinel-2 false color / RGB composite"""
        return {
            'bands': ['B8', 'B4', 'B3'],
            'min': 0,
            'max': 3000,
            'gamma': 1.4
        }
