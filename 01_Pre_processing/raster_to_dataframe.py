# import rasterio
# import numpy as np
# import pandas as pd
# import os
# import re
# from datetime import datetime

# def get_season(month):
#     """Return season string based on month number."""
#     if month in [12, 1, 2]:
#         return 'Winter'
#     elif month in [3, 4, 5]:
#         return 'Spring'
#     elif month in [6, 7, 8]:
#         return 'Summer'
#     else:
#         return 'Fall'

# def raster_to_dataframe(raster_folder, landcover_tif_path, output_csv_path):
#     all_dfs = []

#     # Get all .tif files in the folder (excluding the landcover file)
#     tif_files = [
#         os.path.join(raster_folder, f)
#         for f in os.listdir(raster_folder)
#         if f.lower().endswith(".tif") and os.path.join(raster_folder, f) != landcover_tif_path
#     ]

#     if not tif_files:
#         print("No raster files found.")
#         return

#     # Load landcover data
#     with rasterio.open(landcover_tif_path) as lc_src:
#         lc_data = lc_src.read(1)
#         lc_transform = lc_src.transform

#     for file in tif_files:
#         with rasterio.open(file) as src:
#             height, width = src.height, src.width
#             transform = src.transform

#             # Create grid of pixel coordinates
#             rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
#             rows_flat = rows.flatten()
#             cols_flat = cols.flatten()

#             # Convert pixel coordinates to geographic coordinates
#             xs, ys = rasterio.transform.xy(transform, rows_flat, cols_flat)
#             xs = np.array(xs)
#             ys = np.array(ys)

#             # Read raster bands
#             bands_data = [src.read(i).flatten() for i in range(1, src.count + 1)]
#             band_names = [
#                 desc if desc else f"band_{i}"
#                 for i, desc in enumerate(src.descriptions, start=1)
#             ]

#             df = pd.DataFrame(np.array(bands_data).T, columns=band_names)
#             df["longitude"] = xs
#             df["latitude"] = ys

#             # Extract date and season
#             filename = os.path.basename(file)
#             date_match = re.search(r'\d{4}[-_]\d{2}[-_]\d{2}|\d{8}', filename)
#             if date_match:
#                 date_str = date_match.group(0).replace('_', '-')
#                 try:
#                     date = datetime.strptime(date_str, '%Y-%m-%d').date()
#                 except ValueError:
#                     date = datetime.strptime(date_str, '%Y%m%d').date()
#                 df["date"] = date
#                 df["season"] = get_season(date.month)
#             else:
#                 df["season"] = None
#                 df["date"] = None

#             # Map landcover values using landcover transform
#             lc_rows, lc_cols = rasterio.transform.rowcol(lc_transform, xs, ys)
#             lc_rows = np.clip(lc_rows, 0, lc_data.shape[0] - 1)
#             lc_cols = np.clip(lc_cols, 0, lc_data.shape[1] - 1)
#             landcover_values = lc_data[lc_rows, lc_cols]
#             df["landcover"] = landcover_values

#             df.dropna(inplace=True)
#             all_dfs.append(df)

#     # Combine all into one DataFrame
#     combined_df = pd.concat(all_dfs, ignore_index=True)
#     combined_df.dropna(inplace=True)
#     combined_df.to_csv(output_csv_path, index=False)
#     print(f"Combined data saved to: {output_csv_path}")

import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import pandas as pd
import os
import re
from datetime import datetime

def get_season(month):
    """Return season string based on month number."""
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

def raster_to_dataframe(raster_folder, landcover_tif_path, output_csv_path):
    all_dfs = []

    # Get all .tif files in the folder (excluding the landcover file)
    tif_files = [
        os.path.join(raster_folder, f)
        for f in os.listdir(raster_folder)
        if f.lower().endswith(".tif") and os.path.join(raster_folder, f) != landcover_tif_path
    ]

    if not tif_files:
        print("No raster files found.")
        return

    for file in tif_files:
        with rasterio.open(file) as src:
            height, width = src.height, src.width
            transform = src.transform
            dst_crs = src.crs

            # Reproject landcover to match this raster
            with rasterio.open(landcover_tif_path) as lc_src:
                lc_data_reproj = np.empty((height, width), dtype=lc_src.meta['dtype'])
                reproject(
                    source=rasterio.band(lc_src, 1),
                    destination=lc_data_reproj,
                    src_transform=lc_src.transform,
                    src_crs=lc_src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest  # keep landcover values categorical
                )

            # Create grid of pixel coordinates
            rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
            rows_flat = rows.flatten()
            cols_flat = cols.flatten()

            # Convert pixel coordinates to geographic coordinates
            xs, ys = rasterio.transform.xy(transform, rows_flat, cols_flat)
            xs = np.array(xs)
            ys = np.array(ys)

            # Read raster bands
            bands_data = [src.read(i).flatten() for i in range(1, src.count + 1)]
            band_names = [
                desc if desc else f"band_{i}"
                for i, desc in enumerate(src.descriptions, start=1)
            ]

            df = pd.DataFrame(np.array(bands_data).T, columns=band_names)
            df["longitude"] = xs
            df["latitude"] = ys

            # Extract date and season
            filename = os.path.basename(file)
            date_match = re.search(r'\d{4}[-_]\d{2}[-_]\d{2}|\d{8}', filename)
            if date_match:
                date_str = date_match.group(0).replace('_', '-')
                try:
                    date = datetime.strptime(date_str, '%Y-%m-%d').date()
                except ValueError:
                    date = datetime.strptime(date_str, '%Y%m%d').date()
                df["date"] = date
                df["season"] = get_season(date.month)
            else:
                df["season"] = None
                df["date"] = None

            # Map reprojected landcover values using pixel indices
            lc_rows, lc_cols = rasterio.transform.rowcol(transform, xs, ys)
            lc_rows = np.clip(lc_rows, 0, lc_data_reproj.shape[0] - 1)
            lc_cols = np.clip(lc_cols, 0, lc_data_reproj.shape[1] - 1)
            df["landcover"] = lc_data_reproj[lc_rows, lc_cols]

            df.dropna(inplace=True)
            all_dfs.append(df)

    # Combine all into one DataFrame
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.dropna(inplace=True)
    combined_df.to_csv(output_csv_path, index=False)
    print(f"Combined data saved to: {output_csv_path}")
