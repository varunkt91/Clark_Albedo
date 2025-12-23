# export_manager.py

import ee
import time
import csv
from enum import Enum
from datetime import datetime

from modis_projection import ModisReprojector


# =====================================================
# Export modes
# =====================================================
class ExportMode(Enum):
    MERGED_MODIS = "merged_modis_500m"
    S2_10M = "s2_10m"
    MODIS_ONLY = "modis_only"


# =====================================================
# Export Manager
# =====================================================
class GEEExportManager:
    """
    Export manager for Sentinel-2 / MODIS products with:
    - task throttling
    - robust failure logging (CSV)
    - support for multiple export modes
    """

    def __init__(
        self,
        collection,
        export_folder,
        export_mode: ExportMode,
        export_limit=5000,
        max_active_tasks=15,
        sleep_time=5
    ):
        self.collection = collection
        self.export_folder = export_folder
        self.export_mode = export_mode
        self.export_limit = export_limit
        self.max_active_tasks = max_active_tasks
        self.sleep_time = sleep_time

        self.image_list = self.collection.toList(self.export_limit)

        # -------------------------------------------------
        # Failure logging setup
        # -------------------------------------------------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"export_failures_{export_mode.value}_{timestamp}.csv"
        self._init_log()

    # -------------------------------------------------
    # CSV log initialization
    # -------------------------------------------------
    def _init_log(self):
        with open(self.log_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "index",
                "date",
                "image_id",
                "export_mode",
                "stage",
                "error_message"
            ])

    def _log_failure(self, idx, date, img_id, stage, error):
        with open(self.log_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                idx,
                date,
                img_id,
                self.export_mode.value,
                stage,
                str(error)
            ])

    # -------------------------------------------------
    # Task throttling
    # -------------------------------------------------
    def throttle(self):
        while True:
            active = sum(
                1 for t in ee.batch.Task.list()
                if t.active()
            )

            print(f"Active tasks: {active}/{self.max_active_tasks}", end="\r")

            if active < self.max_active_tasks:
                break

            time.sleep(self.sleep_time)

    # -------------------------------------------------
    # Metadata extraction
    # -------------------------------------------------
    @staticmethod
    def _get_metadata(img, idx):
        try:
            date = ee.Date(img.get("system:time_start")) \
                .format("YYYY-MM-dd").getInfo()
            img_id = img.id().getInfo().replace("/", "_")
            return date, img_id
        except ee.EEException:
            return None, None

    # -------------------------------------------------
    # Band preparation helpers
    # -------------------------------------------------
    @staticmethod
    def _prepare_s2(img):
        return img.toFloat().select([
            'B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12',
            'ndvi','evi','lswi','srti','ndti','crci','mcrc',
            'savi','ndsvi','ndsi','hillshade'
        ])

    @staticmethod
    def _prepare_modis(img):
        return img.toFloat().select(
            'MODIS_Albedo_WSA_shortwave'
        )

    # -------------------------------------------------
    # Prepare export image
    # -------------------------------------------------
    def _prepare_export_image(self, image):

        if self.export_mode == ExportMode.S2_10M:
            export_img = self._prepare_s2(image)
            tag = "S2_10m"

        elif self.export_mode == ExportMode.MERGED_MODIS:
            s2 = image.select([
                'B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12',
                'ndvi','evi','lswi','srti','ndti','crci','mcrc',
                'savi','ndsvi','ndsi','hillshade',
                'MODIS_Albedo_WSA_shortwave'
            ]).toFloat()

            modis = image.select(
                'MODIS_Albedo_WSA_shortwave'
            ).toFloat()

            export_img = ModisReprojector.s2_to_modis_500m(
                s2, modis
            )
            tag = "MERGED_500m"

        elif self.export_mode == ExportMode.MODIS_ONLY:
            export_img = self._prepare_modis(image)
            tag = "MODIS"

        else:
            raise ValueError(f"Unsupported export mode: {self.export_mode}")

        return export_img, tag

    # -------------------------------------------------
    # Main execution loop
    # -------------------------------------------------
    def run(self):

        print(f"\n🚀 Export mode: {self.export_mode.value}")
        print(f"📝 Failure log: {self.log_file}\n")

        size = self.collection.size().getInfo()
        n = min(size, self.export_limit)

        for i in range(n):
            image = ee.Image(self.image_list.get(i))

            # -----------------------------
            # Metadata
            # -----------------------------
            date, img_id = self._get_metadata(image, i)
            if date is None:
                print(f"⚠ Skipping image {i}: metadata missing")
                self._log_failure(
                    idx=i,
                    date=None,
                    img_id=None,
                    stage="metadata",
                    error="Missing system:time_start or image ID"
                )
                continue

            # -----------------------------
            # Prepare export image
            # -----------------------------
            try:
                export_img, tag = self._prepare_export_image(image)
            except Exception as e:
                print(f"⚠ Failed preparing image {i}")
                self._log_failure(
                    idx=i,
                    date=date,
                    img_id=img_id,
                    stage="prepare_export_image",
                    error=e
                )
                continue

            desc = f"{tag}_{date}_{img_id}"
            print(f"\n🔹 Exporting {i+1}/{n}: {desc}")

            # -----------------------------
            # Throttle tasks
            # -----------------------------
            self.throttle()

            # -----------------------------
            # Start export task
            # -----------------------------
            task = ee.batch.Export.image.toDrive(
                image=export_img,
                description=desc,
                folder=self.export_folder,
                fileNamePrefix=desc,
                maxPixels=1e13,
                fileFormat="GeoTIFF"
            )

            try:
                task.start()
                print(f"✔ Started: {desc}")
            except ee.EEException as e:
                print(f"⚠ Failed: {desc}")
                self._log_failure(
                    idx=i,
                    date=date,
                    img_id=img_id,
                    stage="task_start",
                    error=e
                )

        print("\n✅ Export submission complete.")
        print(f"📄 Failure log saved to: {self.log_file}")
