# High-Resolution Land Surface Albedo Estimation

## Project Objective

The objective of this project is to develop a **novel method for estimating land surface albedo at high spatial resolution (10 m)**. We leverage existing **Sentinel-2 imagery (10 m resolution)** and the **MODIS albedo product (500 m resolution)** to build a predictive relationship using machine learning models. Once this relationship is established, we validate the model and **downscale the MODIS albedo to 10 m resolution**. This approach allows us to capture fine-scale spatial variations in albedo, which are not detectable at coarser resolutions. Applications include **urban heat island studies, renewable energy planning, and agricultural monitoring**.

## Framework Overview

The framework is organized into **three modules**, which are also reflected in the GitHub repository:

### Module 1: Data Engineering

This module handles all **data preprocessing and management**:

* Download and process Sentinel-2 images for a specific region and time period.
* Apply **cloud filtering and cloud probability masks** to ensure data quality.
* **Merge Sentinel-2 and MODIS images** based on common acquisition dates.
* Perform **spatial preprocessing**:

  * Clip all images to the **same extent**
  * Mask out invalid pixels
  * Reproject images to a **common resolution**
  * Calculate **spectral and topographic indices**
  * Stack all bands for each Sentinel-2 and MODIS image
* Export processed images as **GeoTIFF files** for downstream use.
* Log any **failed image extractions**.

**Outputs:**

* Preprocessed Sentinel-2 and MODIS images at **common dates and resolutions (500 m)** stored in Google Drive.
* **High-resolution Sentinel-2 band stacks and indices (10 m)** for use in Module 3.

### Module 2: Data Preparation for Machine Learning

This module converts preprocessed imagery into a format suitable for machine learning and deep learning models:

* Convert all images into **Pandas DataFrames** and save as **CSV files**.
* Extract additional information for each pixel, such as:

  * Land cover class
  * Season and month
* These features are used for **data sampling strategies** in Module 3 to ensure proper training, validation, and testing.

### Module 3: Modeling and Albedo Prediction

This module implements **model training and high-resolution albedo prediction**:

* Perform **data sampling and partitioning** into training, validation, and test sets.
* Train **machine learning and deep learning models**, including **hyperparameter tuning**.
* Save trained models and validate their performance.
* Use the trained model and **high-resolution Sentinel-2 bands** from Module 1 to **predict albedo at 10 m resolution**.

## Key Innovation

The main innovation of this project lies in **downscaling coarse-resolution MODIS albedo using Sentinel-2 imagery through machine learning**, capturing fine-scale spatial variability that is otherwise missed at 500 m resolution. This framework integrates **data engineering, feature extraction, and predictive modeling** into a reproducible pipeline suitable for various environmental applications.
