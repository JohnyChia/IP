import os

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np



def save_comparison_figure(filename, images_with_titles):

    fig, axes = plt.subplots(1, len(images_with_titles), figsize=(5 * len(images_with_titles), 5))

    if len(images_with_titles) == 1: axes = [axes]

    

    for ax, (img, title) in zip(axes, images_with_titles):

        if len(img.shape) == 3: ax.imshow(img[..., ::-1])

        else: ax.imshow(img, cmap='gray')

        ax.set_title(title, fontsize=14)

        ax.axis('off')

        

    plt.tight_layout()

    plt.savefig(filename, dpi=150)

    plt.close()



def plot_single_bar_chart(df, x_col, y_col, title, ylabel, out_path):

    plt.figure(figsize=(8, 6))

    ax = sns.barplot(data=df, x=x_col, y=y_col, palette='Set2')

    plt.title(title, fontsize=14)

    plt.ylabel(ylabel, fontsize=12)

    plt.xlabel("")

    

    for container in ax.containers:

                                            

        labels = [f'{v.get_height():.4f}' if v.get_height() > 0 else '' for v in container]

        ax.bar_label(container, labels=labels, padding=3, fontsize=10)

        

    plt.tight_layout()

    plt.savefig(out_path, dpi=150)

    plt.close()



def plot_grouped_bar_chart(df, x_col, metrics, metric_labels, title, ylabel, out_path):

    df_melted = df.melt(id_vars=[x_col], value_vars=metrics, var_name='Metric', value_name='Score')

    df_melted['Metric'] = df_melted['Metric'].map(dict(zip(metrics, metric_labels)))

    

    plt.figure(figsize=(10, 6))

    ax = sns.barplot(data=df_melted, x=x_col, y='Score', hue='Metric', palette='Set2')

    plt.title(title, fontsize=14)

    plt.ylabel(ylabel, fontsize=12)

    plt.xlabel("")

    

    for container in ax.containers:

                                            

        labels = [f'{v.get_height():.4f}' if v.get_height() > 0 else '' for v in container]

        ax.bar_label(container, labels=labels, padding=3, fontsize=10)

        

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    plt.savefig(out_path, dpi=150)

    plt.close()



def generate_quantitative_graphs(all_dfs, base_out_dir):

    os.makedirs(os.path.join(base_out_dir, "situation1"), exist_ok=True)

    os.makedirs(os.path.join(base_out_dir, "situation2"), exist_ok=True)

    os.makedirs(os.path.join(base_out_dir, "situation3"), exist_ok=True)

    

    df_s1, df_s2, df_s3 = all_dfs

    

    sns.set_theme(style="whitegrid")



    s1_agg = df_s1.groupby('algorithm').mean(numeric_only=True).reset_index()

    s2_agg = df_s2.groupby('algorithm').mean(numeric_only=True).reset_index()

    s3_agg = df_s3.groupby('algorithm').mean(numeric_only=True).reset_index()



                 

                                                                                      

    plot_single_bar_chart(

        s1_agg, 'algorithm', 'mean_prediction_overlap',

        "Situation 1: Mean Prediction Overlap", "IoU Overlap",

        os.path.join(base_out_dir, "situation1", "s1_prediction_overlap.jpg")

    )

    plot_single_bar_chart(

        s1_agg, 'algorithm', 'processing_time_s',

        "Situation 1: Processing Time", "Seconds",

        os.path.join(base_out_dir, "situation1", "s1_processing_time.jpg")

    )

    

                 

    plot_grouped_bar_chart(

        s2_agg, 'algorithm', 

        ['precision', 'recall', 'f1'], 

        ['Precision', 'Recall', 'F1-Score'], 

        "Situation 2: Detection Performance (IoU >= 0.50)", "Score",

        os.path.join(base_out_dir, "situation2", "s2_detection_performance.jpg")

    )

    plot_single_bar_chart(

        s2_agg, 'algorithm', 'mean_prediction_overlap',

        "Situation 2: Mean Prediction Overlap", "IoU Overlap",

        os.path.join(base_out_dir, "situation2", "s2_prediction_overlap.jpg")

    )

    plot_single_bar_chart(

        s2_agg, 'algorithm', 'processing_time_s',

        "Situation 2: Processing Time", "Seconds",

        os.path.join(base_out_dir, "situation2", "s2_processing_time.jpg")

    )



                 

    plot_grouped_bar_chart(

        s3_agg, 'algorithm', 

        ['circularity', 'solidity'], 

        ['Circularity', 'Solidity'], 

        "Situation 3: Shape Quality Descriptors", "Score",

        os.path.join(base_out_dir, "situation3", "s3_shape_features.jpg")

    )

    plot_single_bar_chart(

        s3_agg, 'algorithm', 'processing_time_s',

        "Situation 3: Processing Time", "Seconds",

        os.path.join(base_out_dir, "situation3", "s3_processing_time.jpg")

    )

