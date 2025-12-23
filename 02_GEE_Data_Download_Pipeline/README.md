# Sentinel-2 & MODIS Data Fusion and Export Workflow

This repository implements an **end-to-end Google Earth Engine (GEE) pipeline** to:

* Process **Sentinel-2** imagery
* Load and process **MODIS** products (e.g., albedo)
* **Join Sentinel-2 and MODIS by acquisition date**
* Merge both datasets into a single image collection
* Visualize results interactively
* Extract image-level metadata to **Pandas / CSV**
* Export imagery to Google Drive in multiple resolutions

The workflow is modular, class-based, and configurable, making it suitable for research-grade remote sensing applications.

It also handles:

* Projection alignment for all images
* Clipping each image to the same extent
* Keeping only valid pixels
* Cloud masking
* Calculation of spectral indices

---

## 1. Requirements

### Software

* Python 3.8+
* Google Earth Engine Python API

### Python Dependencies

```bash
pip install earthengine-api pandas
```

### Google Earth Engine Setup

1. Create a GEE account: [https://earthengine.google.com/](https://earthengine.google.com/)
2. Authenticate once (first run only):

```python
ee.Authenticate()
```

3. Initialize Earth Engine in your script:

```python
ee.Initialize()
```

---

## 2. Repository Structure

```text
.
├── main.py                     # Main execution script
├── config.py                   # AOI, dates, bands, scale, CRS, etc.
├── raster_collection.py        # Sentinel-2 and MODIS loaders & processors, handles projection, clipping, masking, indices
├── Indices.py                  # Sentinel-2 spectral indices
├── merge_S2_MODIS.py           # Image merge logic for S2 + MODIS
├── vis_params.py               # Visualization parameters
├── metadata_manager.py         # Metadata extraction and CSV export
├── export_manager.py           # Export manager for GEE assets
├── Date_format.py              # Date formatting utilities
└── README.md                   # This file
```

---

## 3. Workflow Overview

### Step 1: Initialize Processing Classes

```python
raster_proc = RasterCollection(
    config=__import__('config'),
    indices_class=Sentinel2Indices
)
```

* Loads configuration from `config.py`
* Registers Sentinel-2 spectral indices
* Ensures all images are clipped to AOI, projected consistently, cloud-masked, and valid pixels retained
* Calculates spectral indices

---

### Step 2: Sentinel-2 Processing

```python
s2_masked = raster_proc.process_sentinel2()
```

Visualization:

```python
Map.addLayer(
    s2_masked,
    VisualizationParams.sentinel2_rgb(),
    'Sentinel-2 RGB'
)
```

---

### Step 3: MODIS Processing

```python
modis = raster_proc.load_modis()
```

Visualization:

```python
Map.addLayer(
    modis.mean(),
    VisualizationParams.modis_albedo_raw(),
    'MODIS Albedo (raw)'
)
```

---

### Step 4: Temporal Join (Sentinel-2 + MODIS)

```python
join = ee.Join.inner()
date_filter = ee.Filter.equals(leftField='date', rightField='date')
joined = join.apply(s2_masked, modis, date_filter)
```

---

### Step 5: Merge Image Bands

```python
mergedCollection = ee.ImageCollection(
    joined.map(merge_images)
)
```

Visualization:

```python
Map.addLayer(
    mergedCollection.first(),
    VisualizationParams.sentinel2_rgb(),
    'Merged Collection'
)
```

---

## 4. Metadata Extraction & CSV Export

### Using `metadata_manager.py`

```python
from metadata_manager import GEECollectionMetadataManager

metadata_manager = GEECollectionMetadataManager(
    collection=mergedCollection,
    csv_path="merged_image_metadata.csv"
)

image_data = metadata_manager.run()
```

* Converts ImageCollection → FeatureCollection → Pandas DataFrame → CSV
* Includes `system_time_start` as datetime
* Preview top rows printed automatically

---

## 5. Image Export to Google Drive

```python
from export_manager import GEEExportManager, ExportMode

exporter = GEEExportManager(
    collection=mergedCollection,
    export_folder="Exports_2021",
    export_mode=ExportMode.MERGED_MODIS
)

exporter.run()
```

### Available Export Modes

| Mode                      | Description                          |
| ------------------------- | ------------------------------------ |
| `ExportMode.S2_10M`       | Sentinel-2 export at 10 m resolution |
| `ExportMode.MERGED_MODIS` | Merged S2 + MODIS export at 500 m    |
| `ExportMode.MODIS_ONLY`   | MODIS-only export                    |

---

## 6. Outputs

* **Google Drive exports** (GeoTIFFs)
* **CSV metadata file**
* **Interactive map visualization** (Jupyter / notebook environment)

---

## 7. Notes & Best Practices

* Ensure AOI and date ranges are correctly set in `config.py`
* Large AOIs or long time ranges may hit GEE task limits
* Prefer exporting in batches for long time series
* Metadata extraction is encapsulated in `GEECollectionMetadataManager`
* RasterCollection handles projection, clipping, cloud masking, valid pixels, and spectral indices calculation
* You can reuse the metadata class for any collection with minimal changes

---

## 8. Citation / Acknowledgement

If you use this workflow in publications, please acknowledge:

* Google Earth Engine
* Sentinel-2 (ESA / Copernicus)
* MODIS (NASA)

---

## 9. Contact

For questions, extensions, or collaboration, feel free to reach out.

---

**End of README**
