import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_cv_results(results):
    """
    Plot RMSE and R2 across CV folds.
    """
    folds = list(range(1, len(results['rmse']) + 1))
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(folds, results['rmse'], marker='o', label='Percentage RMSE')
    plt.title('Percentage RMSE per Fold')
    plt.xlabel('Fold')
    plt.ylabel('Percentage RMSE')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(folds, results['r2'], marker='s', color='orange', label='R² Score')
    plt.title('R² Score per Fold')
    plt.xlabel('Fold')
    plt.ylabel('R² Score')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(r'Output/cross_validation_plots.png')
    plt.show()

def plot_feature_importance(importance_df):
    """
    Plot feature importance as a horizontal bar chart.
    """
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.title('Feature Importance from Random Forest')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(r'feature_importance_plot.png')
    plt.show()


def get_land_cover_labels_and_colors():
    """
    Returns two dictionaries:
    - land_cover_labels: Mapping from land cover ID to name.
    - land_cover_colors: Mapping from land cover ID to hex color.
    """
    land_cover_labels = {
        1: "Evergreen Needleleaf Forests", 2: "Evergreen Broadleaf Forests",
        3: "Deciduous Needleleaf Forests", 4: "Deciduous Broadleaf Forests",
        5: "Mixed Forests", 6: "Closed Shrublands", 7: "Open Shrublands",
        8: "Woody Savannas", 9: "Savannas", 10: "Grasslands",
        11: "Permanent Wetlands", 12: "Croplands", 13: "Urban and Built-up Lands",
        14: "Cropland/Natural Veg. Mosaics", 15: "Permanent Snow and Ice",
        16: "Barren", 17: "Water Bodies"
    }

    land_cover_colors = {
        1: "#05450a", 2: "#086a10", 3: "#54a708", 4: "#78d203", 5: "#009900",
        6: "#c6b044", 7: "#dcd159", 8: "#dade48", 9: "#fbff13", 10: "#b6ff05",
        11: "#27ff87", 12: "#c24f44", 13: "#a5a5a5", 14: "#ff6d4c", 15: "#69fff8",
        16: "#f9ffa4", 17: "#1c0dff"
    }

    return land_cover_labels, land_cover_colors

import pandas as pd
# def plot_class_distribution(train_path, val_path, test_path, class_column="landcover"):
#     # Load CSVs
#     train = pd.read_csv(train_path)
#     val = pd.read_csv(val_path)
#     test = pd.read_csv(test_path)

#     # Count per class
#     train_counts = train[class_column].value_counts().sort_index()
#     val_counts = val[class_column].value_counts().sort_index()
#     test_counts = test[class_column].value_counts().sort_index()

#     # Align indexes
#     all_classes = sorted(set(train_counts.index) | set(val_counts.index) | set(test_counts.index))
#     train_counts = train_counts.reindex(all_classes, fill_value=0)
#     val_counts = val_counts.reindex(all_classes, fill_value=0)
#     test_counts = test_counts.reindex(all_classes, fill_value=0)

#     # Get labels and colors
#     land_cover_labels, land_cover_colors = get_land_cover_labels_and_colors()
#     class_labels = [land_cover_labels.get(c, str(c)) for c in all_classes]
#     class_colors = [land_cover_colors.get(c, "#333333") for c in all_classes]

#     # Plot
#     x = range(len(all_classes))
#     width = 0.25

#     plt.figure(figsize=(14, 7))
#     plt.bar([i - width for i in x], train_counts, width=width, label="Train", color="skyblue")
#     plt.bar(x, val_counts, width=width, label="Validation", color="orange")
#     plt.bar([i + width for i in x], test_counts, width=width, label="Test", color="green")

#     plt.xticks(x, class_labels, rotation=90)
#     plt.xlabel("Land Cover Class")
#     plt.ylabel("Number of Samples")
#     plt.title("Distribution of Samples per Land Cover Class in Train/Val/Test")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(r"plot_r2_per_land_cover.png", dpi=300)
#     plt.show()


import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

def plot_class_distribution(train_path, val_path, test_path, class_column="landcover"):
    # Load CSVs
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)

    # Count per class
    train_counts = train[class_column].value_counts().sort_index()
    val_counts = val[class_column].value_counts().sort_index()
    test_counts = test[class_column].value_counts().sort_index()

    # Align indexes to include all classes
    all_classes = sorted(set(train_counts.index) | set(val_counts.index) | set(test_counts.index))
    train_counts = train_counts.reindex(all_classes, fill_value=0)
    val_counts = val_counts.reindex(all_classes, fill_value=0)
    test_counts = test_counts.reindex(all_classes, fill_value=0)

    # Remove class 0 and classes with 0 samples across all datasets
    mask = (train_counts + val_counts + test_counts) > 0
    mask &= (train_counts.index != 0)
    train_counts = train_counts[mask]
    val_counts = val_counts[mask]
    test_counts = test_counts[mask]
    all_classes = train_counts.index

    # Land cover labels
    land_cover_labels, _ = get_land_cover_labels_and_colors()
    class_labels = [land_cover_labels.get(c, str(c)) for c in all_classes]

    # Bar positions
    x = np.arange(len(all_classes))
    width = 0.25

    # Figure style
    plt.figure(figsize=(16, 8))

    # Plot bars
    bars_train = plt.bar(x - width, train_counts, width=width, color="steelblue", edgecolor='black')
    bars_val   = plt.bar(x, val_counts, width=width, color="orange", edgecolor='black')
    bars_test  = plt.bar(x + width, test_counts, width=width, color="green", edgecolor='black')

    # Total points
    total_train = train_counts.sum()
    total_val = val_counts.sum()
    total_test = test_counts.sum()

    # Custom legend with total points
    plt.legend([
        f"Train ({total_train})",
        f"Validation ({total_val})",
        f"Test ({total_test})"
    ], fontsize=14)

    # Labels and title
    plt.xticks(x, class_labels, rotation=45, ha='right', fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Land Cover Class", fontsize=16)
    plt.ylabel("Number of Samples", fontsize=16)
    plt.title("Distribution of Samples per Land Cover Class in Train/Validation/Test", fontsize=18)

    # Remove gridlines
    plt.grid(False)
    
    plt.tight_layout()
    plt.savefig("plot_class_distribution_professional.png", dpi=600, transparent=True)
    plt.show()



def plot_and_save_land_cover_metrics(data, land_cover_info, output_dir="Output", dataset_name=None):
    """
    Plots RMSE% and R² by land cover type and saves results to files for a given dataset.

    Parameters:
    - data (pd.DataFrame): DataFrame with 'landcover', 'RMSE_percent', and 'R2'.
    - land_cover_info (tuple): Output from get_land_cover_labels_and_colors(), containing:
        - land_cover_labels (dict): Mapping from land cover IDs to names.
        - land_cover_colors (dict): Mapping from land cover IDs to HEX color strings.
    - output_dir (str): Output directory path.
    - dataset_name (str): Name of the dataset (e.g., "train" or "test") for file naming.
    """
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Unpack label & color dicts
    land_cover_labels, land_cover_colors = land_cover_info
    os.makedirs(output_dir, exist_ok=True)

    # ✅ Add labels and colors safely
    data = data.copy()
    data['Land Cover Name'] = data['landcover'].map(land_cover_labels).fillna("Unknown")

    # Build a palette dict (category -> color), with gray fallback
    unique_classes = data['Land Cover Name'].unique()
    palette_dict = {
        lc: land_cover_colors.get(code, "#999999")  # fallback to gray
        for code, lc in zip(data['landcover'], data['Land Cover Name'])
    }

    # --- Plot RMSE% ---
    plt.figure(figsize=(14, 6))
    sns.barplot(
        data=data,
        x='Land Cover Name',
        y='RMSE_percent',
        palette=palette_dict
    )
    plt.xticks(rotation=45, ha='right')
    plt.title(f'RMSE% per Land Cover Type ({dataset_name})')
    plt.ylabel('RMSE (%)')
    plt.xlabel('Land Cover')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/plot_rmse_per_land_cover_{dataset_name}.png", dpi=300)
    plt.close()

    # --- Plot R² ---
    plt.figure(figsize=(14, 6))
    sns.barplot(
        data=data,
        x='Land Cover Name',
        y='R2',
        palette=palette_dict
    )
    plt.xticks(rotation=45, ha='right')
    plt.title(f'R² Score per Land Cover Type ({dataset_name})')
    plt.ylabel('R²')
    plt.xlabel('Land Cover')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/plot_r2_per_land_cover_{dataset_name}.png", dpi=300)
    plt.close()

    # Save CSV
    csv_path = f"{output_dir}/metrics_per_land_cover_{dataset_name}.csv"
    data.to_csv(csv_path, index=False)

    print(f"✅ Plots and CSV for '{dataset_name}' saved to '{output_dir}/'")






# Density plot
def plot_density_per_land_cover(data, output_dir="Output/density_per_land_cover"):
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns

    # ✅ Get labels and colors from helper function
    land_cover_labels, land_cover_colors = get_land_cover_labels_and_colors()

    # ✅ Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # ✅ Loop through land cover classes
    for lc in sorted(data['landcover'].unique()):
        subset = data[data['landcover'] == lc]
        if subset.empty:
            continue

        lc_name = land_cover_labels.get(lc, f"Class {lc}")
        lc_color = land_cover_colors.get(lc, "#333333")

        plt.figure(figsize=(8, 5))
        sns.kdeplot(subset['MODIS_Albedo_WSA_shortwave'], label='Observed', color='black', linewidth=2)
        sns.kdeplot(subset['RF_Predicted'], label='RF_Predicted', color=lc_color, linewidth=2)

        plt.title(f"Albedo Density - {lc_name}")
        plt.xlabel("Albedo")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()

        filename = os.path.join(
            output_dir,
            f"density_lc{lc}_{lc_name.replace(' ', '_').replace('/', '').replace('.', '')}.png"
        )
        plt.savefig(filename, dpi=300)
        plt.close()

    print(f"✅ Density plots saved to '{output_dir}'")


# def plot_density_scatter_per_land_cover(data, output_dir=r"Output/scatter_density_per_land_cover"):
#     # Get labels and colors
#     land_cover_labels, land_cover_colors = get_land_cover_labels_and_colors()

#     os.makedirs(output_dir, exist_ok=True)

#     for lc in sorted(data['landcover'].unique()):
#         subset = data[data['landcover'] == lc]
#         if subset.empty:
#             continue

#         lc_name = land_cover_labels.get(lc, f"Class {lc}")
#         lc_color = land_cover_colors.get(lc, "#333333")

#         x = subset['MODIS_Albedo_WSA_shortwave']
#         y = subset['RF_Predicted']

#         # Compute axis limits with padding
#         min_val = min(x.min(), y.min())
#         max_val = max(x.max(), y.max())
#         range_padding = (max_val - min_val) * 0.05  # 5% padding
#         xlim = (min_val - range_padding, max_val + range_padding)
#         ylim = (min_val - range_padding, max_val + range_padding)

#         plt.figure(figsize=(7, 6))
#         sns.kdeplot(
#             x=x,
#             y=y,
#             fill=True,
#             cmap="Accent",
#             thresh=0.05,
#             levels=100
#         )

#         plt.plot([min_val, max_val], [min_val, max_val], '--', color='gray', linewidth=1)

#         plt.title(f"Scatter Density Plot - {lc_name}")
#         plt.xlabel("Observed Albedo")
#         plt.ylabel("Predicted Albedo (RF_Predicted)")
#         plt.xlim(xlim)
#         plt.ylim(ylim)
#         plt.tight_layout()

#         filename = os.path.join(
#             output_dir,
#             f"scatter_density_lc{lc}_{lc_name.replace(' ', '_').replace('/', '').replace('.', '')}.png"
#         )
#         plt.savefig(filename, dpi=300)
#         plt.close()

#     print(f"✅ Density scatter plots saved to '{output_dir}'")

def plot_density_scatter_per_land_cover(
    data,
    metrics_df_per_lc,
    output_dir=r"output/scatter_density_per_land_cover",
    dataset_name="train"
):
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    land_cover_labels, land_cover_colors = get_land_cover_labels_and_colors()
    os.makedirs(output_dir, exist_ok=True)

    for lc in sorted(data['landcover'].unique()):
        subset = data[data['landcover'] == lc][['MODIS_Albedo_WSA_shortwave','RF_Predicted']].dropna()
        if subset.empty:
            continue

        x = subset['MODIS_Albedo_WSA_shortwave']
        y = subset['RF_Predicted']

        # Skip if too few points or zero variance
        if len(subset) < 5 or np.isclose(np.var(x), 0) or np.isclose(np.var(y), 0):
            print(f"⚠️ Skipping land cover {lc} ({land_cover_labels.get(lc, lc)}) due to insufficient variation or points")
            continue

        lc_name = land_cover_labels.get(lc, f"Class {lc}")
        lc_color = land_cover_colors.get(lc, "#333333")

        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())
        range_padding = (max_val - min_val) * 0.05
        xlim = (min_val - range_padding, max_val + range_padding)
        ylim = (min_val - range_padding, max_val + range_padding)

        plt.figure(figsize=(7, 6))

        try:
            # KDE plot
            sns.kdeplot(
                x=x,
                y=y,
                fill=True,
                cmap="Accent",
                thresh=0.05,
                levels=100
            )
        except ValueError:
            # Fallback to scatter if KDE fails
            print(f"⚠️ KDE failed for land cover {lc} ({lc_name}). Using scatter plot instead.")
            plt.scatter(x, y, alpha=0.5, color=lc_color)

        # 1:1 reference line
        plt.plot([min_val, max_val], [min_val, max_val], '--', color='gray', linewidth=1)

        # Add metrics
        if lc in metrics_df_per_lc.index:
            lc_metrics = metrics_df_per_lc.loc[lc]
            metrics_text = "\n".join([f"{k}: {v:.3f}" for k, v in lc_metrics.items()])
            plt.text(
                0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7)
            )

        plt.title(f"Scatter Density Plot - {lc_name} ({dataset_name})")
        plt.xlabel("Observed Albedo")
        plt.ylabel("Predicted Albedo (RF_Predicted)")
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.tight_layout()

        filename = os.path.join(
            output_dir,
            f"scatter_density_{dataset_name}_lc{lc}_{lc_name.replace(' ', '_').replace('/', '').replace('.', '')}.png"
        )
        plt.savefig(filename, dpi=300)
        plt.close()

    print(f"✅ Density scatter plots with metrics for '{dataset_name}' saved to '{output_dir}'")






def plot_scatter_per_land_cover_with_metrics(data, metrics_df_per_lc, 
                                             output_dir="Output/scatter_per_land_cover", 
                                             dataset_name="train"):
    """
    Creates scatter plots of observed vs predicted albedo for each land cover type,
    including regression metrics per land cover.

    Parameters:
    - data (pd.DataFrame): Must contain 'landcover', 'MODIS_Albedo_WSA_shortwave', 'RF_Predicted'
    - metrics_df_per_lc (pd.DataFrame): Metrics per landcover type (index=landcover)
    - output_dir (str): Directory to save plots
    - dataset_name (str): 'train' or 'test' for file naming
    """
    import os
    import matplotlib.pyplot as plt

    land_cover_labels, land_cover_colors = get_land_cover_labels_and_colors()
    os.makedirs(output_dir, exist_ok=True)

    for lc in sorted(data['landcover'].unique()):
        subset = data[data['landcover'] == lc]
        if subset.empty:
            continue

        lc_name = land_cover_labels.get(lc, f"Class {lc}")
        lc_color = land_cover_colors.get(lc, "#333333")

        x = subset['MODIS_Albedo_WSA_shortwave']
        y = subset['RF_Predicted']

        # Compute axis limits with padding
        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())
        range_padding = (max_val - min_val) * 0.05
        xlim = (min_val - range_padding, max_val + range_padding)
        ylim = (min_val - range_padding, max_val + range_padding)

        plt.figure(figsize=(7, 6))
        plt.scatter(x, y, s=10, color=lc_color, alpha=0.6)
        plt.plot([min_val, max_val], [min_val, max_val], '--', color='gray', linewidth=1)  # 1:1 line

        # Add metrics
        if lc in metrics_df_per_lc.index:
            lc_metrics = metrics_df_per_lc.loc[lc]
            metrics_text = "\n".join([f"{k}: {v:.3f}" for k, v in lc_metrics.items()])
            plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
                     verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))

        plt.title(f"Scatter Plot - {lc_name} ({dataset_name})")
        plt.xlabel("Observed Albedo")
        plt.ylabel("Predicted Albedo (RF_Predicted)")
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.tight_layout()

        filename = os.path.join(
            output_dir,
            f"scatter_{dataset_name}_lc{lc}_{lc_name.replace(' ', '_').replace('/', '').replace('.', '')}.png"
        )
        plt.savefig(filename, dpi=300)
        plt.close()

    print(f"✅ Scatter plots with metrics for '{dataset_name}' saved to '{output_dir}'")

# import os
# import matplotlib.pyplot as plt
# import seaborn as sns
# from math import ceil
# import string

# def plot_scatter_per_land_cover_with_metrics(
#         data, 
#         metrics_df_per_lc, 
#         output_dir="output/scatter_per_land_cover", 
#         dataset_name="train",
#         n_cols=2,
#         xlabel="Observed Albedo",
#         ylabel="Predicted Albedo",
#         title_fontsize=18,
#         label_fontsize=16,
#         tick_fontsize=14,
#         metrics_fontsize=12,
#         panel_label=True):
#     """
#     Creates a single publication-quality figure with scatter plots of observed vs predicted albedo
#     for each land cover type, including regression metrics and optional panel labels (a, b, c, ...).
    
#     Parameters:
#     - data (pd.DataFrame): Must contain 'landcover', 'MODIS_Albedo_WSA_shortwave', 'RF_Predicted'
#     - metrics_df_per_lc (pd.DataFrame): Metrics per landcover type (index=landcover)
#     - output_dir (str): Directory to save plot
#     - dataset_name (str): 'train' or 'test' for file naming
#     - n_cols (int): Number of columns of subplots
#     - xlabel, ylabel (str): Axis labels
#     - title_fontsize (int): Font size for subplot titles
#     - label_fontsize (int): Font size for axis labels
#     - tick_fontsize (int): Font size for ticks
#     - metrics_fontsize (int): Font size for metrics text
#     - panel_label (bool): If True, add a), b), c), ... labels to subplots
#     """
    
#     land_cover_labels, land_cover_colors = get_land_cover_labels_and_colors()
#     os.makedirs(output_dir, exist_ok=True)
    
#     landcover_list = sorted(data['landcover'].unique())
#     n_landcovers = len(landcover_list)
#     n_rows = ceil(n_landcovers / n_cols)
    
#     # Seaborn style for publication
#     sns.set_style("whitegrid")
#     sns.set_context("talk", font_scale=1.2)
    
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows), squeeze=False)
    
#     # Panel labels: a, b, c, ...
#     labels = list(string.ascii_lowercase)
    
#     for idx, lc in enumerate(landcover_list):
#         row = idx // n_cols
#         col = idx % n_cols
#         ax = axes[row, col]
        
#         subset = data[data['landcover'] == lc]
#         if subset.empty:
#             ax.axis('off')
#             continue
        
#         lc_name = land_cover_labels.get(lc, f"Class {lc}")
#         lc_color = land_cover_colors.get(lc, "#333333")
        
#         x = subset['MODIS_Albedo_WSA_shortwave']
#         y = subset['RF_Predicted']
        
#         # Axis limits with padding
#         min_val = min(x.min(), y.min())
#         max_val = max(x.max(), y.max())
#         range_padding = (max_val - min_val) * 0.05
#         xlim = (min_val - range_padding, max_val + range_padding)
#         ylim = (min_val - range_padding, max_val + range_padding)
        
#         # Scatter plot
#         sns.scatterplot(x=x, y=y, ax=ax, color=lc_color, s=50, alpha=0.6)
#         ax.plot([min_val, max_val], [min_val, max_val], '--', color='gray', linewidth=1)
#         ax.set_xlim(xlim)
#         ax.set_ylim(ylim)
#         ax.set_xlabel(xlabel, fontsize=label_fontsize)
#         ax.set_ylabel(ylabel, fontsize=label_fontsize)
#         ax.set_title(f"{lc_name} ({dataset_name})", fontsize=title_fontsize)
#         ax.tick_params(axis='both', labelsize=tick_fontsize)
        
#         # Metrics text
#         if lc in metrics_df_per_lc.index:
#             lc_metrics = metrics_df_per_lc.loc[lc]
#             metrics_text = "\n".join([f"{k}: {v:.3f}" for k, v in lc_metrics.items()])
#             ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
#                     verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7),
#                     fontsize=metrics_fontsize)
        
#         # Panel label
#         if panel_label and idx < len(labels):
#             ax.text(-0.15, 1.05, f"{labels[idx]})", transform=ax.transAxes,
#                     fontsize=title_fontsize, fontweight='bold', va='top', ha='right')
    
#     # Hide empty subplots
#     for empty_idx in range(n_landcovers, n_rows*n_cols):
#         row = empty_idx // n_cols
#         col = empty_idx % n_cols
#         axes[row, col].axis('off')
    
#     plt.tight_layout()
#     filename = os.path.join(output_dir, f"scatter_{dataset_name}_all_landcovers.png")
#     plt.savefig(filename, dpi=300, bbox_inches='tight')
#     plt.close()
    
#     print(f"✅ Publication-ready scatter plot figure saved as '{filename}'")





import os
import matplotlib.pyplot as plt

def plot_predicted_vs_actual(y_train, y_train_pred,
                             y_test, y_test_pred,
                             y_test_2020, y_test_2020_pred,
                             metrics_df,
                             output_dir="output",
                             figsize=(20, 6),
                             dpi=300,
                             font_size=20,
                             marker_size=30,
                             axis_tick_size=18):
    """
    Plot Predicted vs Actual for Train, Test, and Test_2020 sets with regression metrics.
    Publication-ready figure with control over font size, marker size, axis tick size, and high resolution.
    """

    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=figsize)

    # Common properties
    scatter_kwargs = dict(alpha=0.6, s=marker_size, edgecolor='k')
    line_kwargs = dict(color='red', linestyle='--', linewidth=2)
    title_font = {'fontsize': font_size + 2, 'fontweight': 'bold'}
    label_font = {'fontsize': font_size}

    # --------------------------
    # Train plot
    # --------------------------
    plt.subplot(1, 3, 1)
    plt.scatter(y_train, y_train_pred, **scatter_kwargs)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], **line_kwargs)
    plt.xlabel("MODIS Albedo", **label_font)
    plt.ylabel("Estimated", **label_font)
    plt.title("Train 2021 (In-year)", **title_font)
    plt.xticks(fontsize=axis_tick_size)
    plt.yticks(fontsize=axis_tick_size)
    train_text = "\n".join([f"{k}: {v:.3f}" for k, v in metrics_df.loc["Train"].items()])
    plt.text(0.05, 0.95, train_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7), fontsize=font_size)

    # --------------------------
    # Test plot
    # --------------------------
    plt.subplot(1, 3, 2)
    plt.scatter(y_test, y_test_pred, **scatter_kwargs)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], **line_kwargs)
    plt.xlabel("MODIS Albedo", **label_font)
    plt.ylabel("Estimated", **label_font)
    plt.title("Test 2021 (In-year)", **title_font)
    plt.xticks(fontsize=axis_tick_size)
    plt.yticks(fontsize=axis_tick_size)
    test_text = "\n".join([f"{k}: {v:.3f}" for k, v in metrics_df.loc["Test_All"].items()])
    plt.text(0.05, 0.95, test_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7), fontsize=font_size)

    # --------------------------
    # Test 2020 plot
    # --------------------------
    plt.subplot(1, 3, 3)
    plt.scatter(y_test_2020, y_test_2020_pred, **scatter_kwargs)
    plt.plot([y_test_2020.min(), y_test_2020.max()], [y_test_2020.min(), y_test_2020.max()], **line_kwargs)
    plt.xlabel("MODIS Albedo", **label_font)
    plt.ylabel("Estimated", **label_font)
    plt.title("Test 2020 (Out-of-year)", **title_font)
    plt.xticks(fontsize=axis_tick_size)
    plt.yticks(fontsize=axis_tick_size)
    test2020_text = "\n".join([f"{k}: {v:.3f}" for k, v in metrics_df.loc["Test_2020"].items()])
    plt.text(0.05, 0.95, test2020_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7), fontsize=font_size)

    plt.tight_layout()
    pred_plot_file = os.path.join(output_dir, "predicted_vs_actual.png")
    plt.savefig(pred_plot_file, dpi=dpi, bbox_inches='tight')
    plt.show()

    print(f"✅ Publication-ready predicted vs actual plots saved to {pred_plot_file}")
