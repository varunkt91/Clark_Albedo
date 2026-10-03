"""Export MODIS albedo (MCD43A3) stacked with LST (MOD11A1) for Delhi to Google Drive.

Usage:
    pip install earthengine-api
    earthengine authenticate
    python Download_MODIS_Albedo_LST_Delhi.py --project YOUR_GEE_PROJECT [--freq monthly|daily]
"""
import argparse
from datetime import date

import ee

# Delhi NCT approximate bounding box (lon_min, lat_min, lon_max, lat_max)
DELHI_BBOX = [76.84, 28.40, 77.35, 28.88]
START_YEAR = 2010
SCALE = 500  # m; MODIS albedo native resolution
DRIVE_FOLDER = "MODIS_Albedo_LST_Delhi"

ALBEDO_BANDS = [
    "Albedo_BSA_Band1", "Albedo_BSA_Band2", "Albedo_BSA_Band3", "Albedo_BSA_Band4",
    "Albedo_BSA_Band5", "Albedo_BSA_Band6", "Albedo_BSA_Band7",
    "Albedo_BSA_vis", "Albedo_BSA_nir", "Albedo_BSA_shortwave",
    "Albedo_WSA_Band1", "Albedo_WSA_Band2", "Albedo_WSA_Band3", "Albedo_WSA_Band4",
    "Albedo_WSA_Band5", "Albedo_WSA_Band6", "Albedo_WSA_Band7",
    "Albedo_WSA_vis", "Albedo_WSA_nir", "Albedo_WSA_shortwave",
]
QA_BANDS = ["BRDF_Albedo_Band_Mandatory_Quality_shortwave"]
LST_BANDS = ["LST_Day_1km", "LST_Night_1km", "QC_Day", "QC_Night",
             "Day_view_time", "Night_view_time", "Emis_31", "Emis_32"]


def prepare_albedo(img):
    """Apply 0.001 scale factor to albedo bands; keep quality band unscaled."""
    scaled = img.select(ALBEDO_BANDS).multiply(0.001).toFloat()
    return scaled.addBands(img.select(QA_BANDS).toFloat()).copyProperties(img, ["system:time_start"])


def prepare_lst(img):
    """Convert LST to Kelvin/Celsius (scale 0.02) and keep other bands."""
    lst = img.select(["LST_Day_1km", "LST_Night_1km"]).multiply(0.02).toFloat()
    lst_c = lst.subtract(273.15).rename(["LST_Day_C", "LST_Night_C"])
    emis = img.select(["Emis_31", "Emis_32"]).multiply(0.002).add(0.49).toFloat()
    other = img.select(["QC_Day", "QC_Night", "Day_view_time", "Night_view_time"]).toFloat()
    return (lst.rename(["LST_Day_K", "LST_Night_K"]).addBands([lst_c, emis, other])
            .copyProperties(img, ["system:time_start"]))


def stack_for_range(start, end, region):
    """Mean composite of albedo + LST bands over [start, end)."""
    alb = (ee.ImageCollection("MODIS/061/MCD43A3").filterDate(start, end)
           .filterBounds(region).map(prepare_albedo).mean())
    lst = (ee.ImageCollection("MODIS/061/MOD11A1").filterDate(start, end)
           .filterBounds(region).map(prepare_lst).mean())
    # LST (1 km) is resampled onto the 500 m albedo grid
    proj = ee.ImageCollection("MODIS/061/MCD43A3").first().select(0).projection()
    lst = lst.resample("bilinear").reproject(proj)
    return alb.addBands(lst).clip(region)


def periods(freq):
    today = date.today()
    for y in range(START_YEAR, today.year + 1):
        if freq == "monthly":
            for m in range(1, 13):
                s = date(y, m, 1)
                if s > today:
                    return
                e = date(y + (m == 12), m % 12 + 1, 1)
                yield s.isoformat(), e.isoformat(), f"{y}_{m:02d}"
        else:
            from datetime import timedelta
            d = date(y, 1, 1)
            while d.year == y and d <= today:
                n = d + timedelta(days=1)
                yield d.isoformat(), n.isoformat(), d.strftime("%Y_%m_%d")
                d = n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True, help="Google Cloud project registered for Earth Engine")
    p.add_argument("--freq", choices=["monthly", "daily"], default="monthly")
    p.add_argument("--folder", default=DRIVE_FOLDER)
    a = p.parse_args()

    ee.Initialize(project=a.project)
    region = ee.Geometry.Rectangle(DELHI_BBOX)

    count = 0
    for start, end, tag in periods(a.freq):
        img = stack_for_range(start, end, region)
        task = ee.batch.Export.image.toDrive(
            image=img, description=f"Delhi_Albedo_LST_{tag}", folder=a.folder,
            fileNamePrefix=f"Delhi_Albedo_LST_{tag}", region=region,
            scale=SCALE, crs="EPSG:4326", maxPixels=1e9)
        task.start()
        count += 1
    print(f"Started {count} export tasks. Monitor at https://code.earthengine.google.com/tasks")


if __name__ == "__main__":
    main()
