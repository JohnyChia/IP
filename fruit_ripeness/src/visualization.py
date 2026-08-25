import os

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd



def plot_distribution(df, value_col, class_col, title, ylabel, out_path):

                                      

    df_clean = df.dropna(subset=[value_col, class_col])

    if df_clean.empty:

        return

        

    plt.figure(figsize=(8, 6))

    sns.boxplot(data=df_clean, x=class_col, y=value_col, hue=class_col, palette='Set2', legend=False)

    sns.stripplot(data=df_clean, x=class_col, y=value_col, color='black', alpha=0.5, jitter=True)

    plt.title(title, fontsize=14)

    plt.ylabel(ylabel, fontsize=12)

    plt.xlabel("Predicted Ripeness Class")

    plt.tight_layout()

    plt.savefig(out_path, dpi=150)

    plt.close()



def plot_processing_time(hsv_df, lab_df, out_path):

    plt.figure(figsize=(8, 6))

    

    hsv_time = hsv_df.copy()

    hsv_time['algorithm'] = 'HSV Algorithm'

    lab_time = lab_df.copy()

    lab_time['algorithm'] = 'LAB Algorithm'

    

    df = pd.concat([hsv_time, lab_time], ignore_index=True)

    

    if df.empty:

        return

        

    sns.barplot(data=df, x='algorithm', y='processing_time_s', palette='Set2', errorbar=None, hue='algorithm', legend=False)

    plt.title("Algorithm Processing Time Comparison", fontsize=14)

    plt.ylabel("Processing Time (s)", fontsize=12)

    plt.xlabel("")

    plt.tight_layout()

    plt.savefig(out_path, dpi=150)

    plt.close()



def plot_classification_comparison(hsv_df, lab_df, out_path):

    plt.figure(figsize=(10, 6))

    

    hsv_counts = hsv_df['predicted_ripeness'].value_counts().reset_index()

    hsv_counts.columns = ['predicted_ripeness_class', 'count']

    hsv_counts['algorithm'] = 'HSV Algorithm'

    

    lab_counts = lab_df['predicted_ripeness'].value_counts().reset_index()

    lab_counts.columns = ['predicted_ripeness_class', 'count']

    lab_counts['algorithm'] = 'LAB Algorithm'

    

    counts = pd.concat([hsv_counts, lab_counts], ignore_index=True)

    if counts.empty:

        return

        

    sns.barplot(data=counts, x='algorithm', y='count', hue='predicted_ripeness_class', palette='Set2')

    plt.title("Heuristic Classification Results by Algorithm", fontsize=14)

    plt.ylabel("Number of Images", fontsize=12)

    plt.xlabel("")

    plt.legend(title="Ripeness Class")

    plt.tight_layout()

    plt.savefig(out_path, dpi=150)

    plt.close()



def generate_graphs(hsv_csv_path, lab_csv_path, out_dir):

    hsv_df = pd.read_csv(hsv_csv_path)

    lab_df = pd.read_csv(lab_csv_path)

    

    os.makedirs(os.path.join(out_dir, "hsv"), exist_ok=True)

    os.makedirs(os.path.join(out_dir, "lab"), exist_ok=True)

    os.makedirs(os.path.join(out_dir, "controlled_lighting"), exist_ok=True)

    

                

    if not hsv_df.empty:

        plot_distribution(hsv_df, 'mean_hue', 'predicted_ripeness', 

                         "HSV: Mean Hue Distribution", "Mean Hue (0-179)",

                         os.path.join(out_dir, "hsv", "hue_distribution.jpg"))

        plot_distribution(hsv_df, 'mean_saturation', 'predicted_ripeness', 

                         "HSV: Mean Saturation Distribution", "Mean Saturation (0-255)",

                         os.path.join(out_dir, "hsv", "saturation_distribution.jpg"))

        plot_distribution(hsv_df, 'mean_value', 'predicted_ripeness', 

                         "HSV: Mean Value Distribution", "Mean Value (0-255)",

                         os.path.join(out_dir, "hsv", "value_distribution.jpg"))

                         

                

    if not lab_df.empty:

        plot_distribution(lab_df, 'mean_L', 'predicted_ripeness', 

                         "LAB: Mean L* Distribution", "Mean L* (0-255)",

                         os.path.join(out_dir, "lab", "L_distribution.jpg"))

        plot_distribution(lab_df, 'mean_a', 'predicted_ripeness', 

                         "LAB: Mean a* Distribution", "Mean a* (0-255)",

                         os.path.join(out_dir, "lab", "a_distribution.jpg"))

        plot_distribution(lab_df, 'mean_b', 'predicted_ripeness', 

                         "LAB: Mean b* Distribution", "Mean b* (0-255)",

                         os.path.join(out_dir, "lab", "b_distribution.jpg"))

                         

                

    plot_classification_comparison(hsv_df, lab_df, os.path.join(out_dir, "controlled_lighting", "classification_comparison.jpg"))

    plot_processing_time(hsv_df, lab_df, os.path.join(out_dir, "controlled_lighting", "processing_time_comparison.jpg"))

