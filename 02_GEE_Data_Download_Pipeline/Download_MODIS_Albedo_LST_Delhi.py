"""Daily Delhi stack exported to Google Drive on the native MODIS 500 m grid.

Bands per day (on the MODIS 500 m sinusoidal grid; continuous bands bilinear, SRTM averaged):
  MODIS albedo (MCD43A3, daily, QA-masked), MODIS LST + emissivity (MOD11A1,
  daily, cloud/QC-masked), MODIS ET/PET (MOD16A2GF, 8-day product -> the
  composite covering that day), ERA5-Land air temp + evaporation (daily),
  elevation / slope / aspect (SRTM 30 m, mean-aggregated to 500 m),
  yearly land cover (MCD12Q1, nearest neighbour; year clamped to available range).

Usage:
    pip install earthengine-api
    earthengine authenticate
    python Download_MODIS_Albedo_LST_Delhi.py --project YOUR_GEE_PROJECT \
        [--start 2010-01-01] [--end 2024-12-31] [--folder NAME]

Dates come from days with MCD43A3 data.
"""
import argparse
import time
from datetime import date, datetime, timedelta, timezone

import ee

DELHI_BBOX = [76.84, 28.40, 77.35, 28.88]  # lon_min, lat_min, lon_max, lat_max
DRIVE_FOLDER = "MODIS_Albedo_LST_ET_Delhi_Daily"
MAX_QUEUED = 2000  # Earth Engine allows ~3000 queued tasks

ALBEDO_BANDS = (
    [f"Albedo_{k}_Band{i}" for k in ("BSA", "WSA") for i in range(1, 8)]
    + [f"Albedo_{k}_{b}" for k in ("BSA", "WSA") for b in ("vis", "nir", "shortwave")]
)
ALBEDO_QA = "BRDF_Albedo_Band_Mandatory_Quality_shortwave"


def masked_template(names):
    """Fully masked float image with the given bands, so empty days keep a fixed band list."""
    return ee.Image.constant([0] * len(names)).rename(names).toFloat().updateMask(0)


def daily_mosaic(col, names):
    return ee.ImageCollection([masked_template(names)]).merge(col).mosaic().select(names)


def albedo_day(start, end, region):
    def prep(img):
        good = img.select(ALBEDO_QA).eq(0)  # 0 = best quality (full BRDF inversion)
        a = img.select(ALBEDO_BANDS).multiply(0.001).toFloat().updateMask(good)
        return a.addBands(img.select(ALBEDO_QA).toFloat().rename("Albedo_QA"))
    col = (ee.ImageCollection("MODIS/061/MCD43A3").filterDate(start, end)
           .filterBounds(region).map(prep))
    return daily_mosaic(col, ALBEDO_BANDS + ["Albedo_QA"])


def lst_day(start, end, region):
    names = ["LST_Day_K", "LST_Night_K", "LST_Day_C", "LST_Night_C",
             "Emis_31", "Emis_32", "Day_view_time", "Night_view_time"]

    def prep(img):
        qd, qn = img.select("QC_Day"), img.select("QC_Night")
        # QC bits 0-1: 00 good, 01 other quality, 10 cloud, 11 not produced -> keep <2
        day_ok = qd.bitwiseAnd(3).lt(2)
        night_ok = qn.bitwiseAnd(3).lt(2)
        day = img.select("LST_Day_1km").multiply(0.02).updateMask(day_ok)
        night = img.select("LST_Night_1km").multiply(0.02).updateMask(night_ok)
        emis = img.select(["Emis_31", "Emis_32"]).multiply(0.002).add(0.49)
        vt = img.select(["Day_view_time", "Night_view_time"]).multiply(0.1)
        return (day.rename("LST_Day_K").addBands(night.rename("LST_Night_K"))
                .addBands(day.subtract(273.15).rename("LST_Day_C"))
                .addBands(night.subtract(273.15).rename("LST_Night_C"))
                .addBands(emis).addBands(vt).toFloat())
    col = (ee.ImageCollection("MODIS/061/MOD11A1").filterDate(start, end)
           .filterBounds(region).map(prep))
    return daily_mosaic(col, names)


def et_day(day, region):
    """MOD16A2GF is 8-day: take the composite whose window contains `day`."""
    names = ["ET_mm", "PET_mm", "LE_Wm2", "PLE_Wm2"]

    def prep(img):
        ok = img.select("ET_QC").bitwiseAnd(1).eq(0)  # bit0 0 = good quality
        out = img.select(["ET", "PET"]).multiply(0.1)  # kg/m2 per 8 days
        le = img.select(["LE", "PLE"]).multiply(10000).divide(86400)  # J/m2/day -> W/m2
        return out.addBands(le).rename(names).toFloat().updateMask(ok)
    d = ee.Date(day)
    col = (ee.ImageCollection("MODIS/061/MOD16A2GF").filterDate(d.advance(-7, "day"), d.advance(1, "day"))
           .filterBounds(region).sort("system:time_start", False).limit(1).map(prep))
    return daily_mosaic(col, names)


def era5_day(start, end):
    names = ["Air_Temp_Mean_C", "Air_Temp_Min_C", "Air_Temp_Max_C",
             "Dewpoint_C", "Total_Evap_mm", "Potential_Evap_mm"]

    def prep(img):
        t = img.select(["temperature_2m", "temperature_2m_min", "temperature_2m_max",
                        "dewpoint_temperature_2m"]).subtract(273.15)
        e = img.select(["total_evaporation_sum", "potential_evaporation_sum"]).multiply(-1000)
        return t.addBands(e).rename(names).toFloat()  # ERA5 evap is negative downward -> mm
    col = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(start, end).map(prep)
    return daily_mosaic(col, names)


def modis_projection():
    """Native MODIS sinusoidal 500 m grid, taken from the albedo product."""
    return ee.ImageCollection("MODIS/061/MCD43A3").first().select(0).projection()


def topo(proj):
    dem = ee.Image("USGS/SRTMGL1_003").rename("Elevation_m")
    t = dem.addBands(ee.Terrain.slope(dem).rename("Slope_deg")).addBands(
        ee.Terrain.aspect(dem).rename("Aspect_deg")).toFloat()
    return t.reduceResolution(ee.Reducer.mean(), maxPixels=4096).reproject(proj)


def landcover(day, proj):
    """Yearly MCD12Q1 land cover (nearest neighbour) for the year of `day`."""
    names = ["LC_Type1_IGBP", "LC_Type2_UMD", "LC_Type3_LAI", "LC_Type4_BGC",
             "LC_Type5_PFT", "LC_Prop1_LCCS", "LC_Prop2_LCCS", "LC_Prop3_LCCS", "LC_QC"]
    col = ee.ImageCollection("MODIS/061/MCD12Q1")
    years = col.aggregate_array("system:time_start")
    t = ee.Date(day).update(month=1, day=1).millis()
    t = ee.Number(t).min(ee.Number(years.reduce(ee.Reducer.max())))  # latest year if beyond range
    img = col.filter(ee.Filter.eq("system:time_start", t)).first()
    lc = img.select(["LC_Type1", "LC_Type2", "LC_Type3", "LC_Type4", "LC_Type5",
                     "LC_Prop1", "LC_Prop2", "LC_Prop3", "QC"]).rename(names)
    return lc.toFloat().reproject(proj)


def build_image(day, region, topo_img, proj):
    start = ee.Date(day)
    end = start.advance(1, "day")
    stack = (albedo_day(start, end, region)
             .addBands(lst_day(start, end, region))
             .addBands(et_day(day, region))
             .addBands(era5_day(start, end)))
    stack = stack.resample("bilinear").addBands(topo_img).addBands(landcover(day, proj))
    return stack.clip(region).set("date", day)


def available_days(start, end, region):
    col = (ee.ImageCollection("MODIS/061/MCD43A3").filterDate(start, end).filterBounds(region))
    ms = col.aggregate_array("system:time_start").getInfo()
    return sorted({datetime.fromtimestamp(m / 1000, timezone.utc).date().isoformat() for m in ms})


def wait_for_queue():
    while True:
        active = [t for t in ee.batch.Task.list() if t.state in ("READY", "RUNNING")]
        if len(active) < MAX_QUEUED:
            return
        time.sleep(300)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True, help="Cloud project registered for Earth Engine")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default=(date.today() + timedelta(days=1)).isoformat())
    p.add_argument("--folder", default=DRIVE_FOLDER)
    a = p.parse_args()

    ee.Initialize(project=a.project)
    region = ee.Geometry.Rectangle(DELHI_BBOX)
    proj = modis_projection()
    topo_img = topo(proj)
    info = proj.getInfo()  # client-side CRS string + 500 m affine transform
    crs, transform = info["wkt"] if "wkt" in info else info["crs"], info["transform"]

    days = available_days(a.start, a.end, region)
    print(f"{len(days)} days with MODIS albedo data")
    for i, day in enumerate(days):
        if i and i % 500 == 0:
            wait_for_queue()
        name = f"Delhi_daily_{day.replace('-', '_')}"
        ee.batch.Export.image.toDrive(
            image=build_image(day, region, topo_img, proj), description=name, folder=a.folder,
            fileNamePrefix=name, region=region, crs=crs, crsTransform=transform,
            maxPixels=1e10).start()
    print(f"Started {len(days)} tasks. Monitor: https://code.earthengine.google.com/tasks")


if __name__ == "__main__":
    main()
