import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os



OUT_CSV_DIR = r"C:\IP\thesis_evaluation\outputs"

GRAPH_DIR = r"C:\IP\thesis_evaluation\graphs"

os.makedirs(OUT_CSV_DIR, exist_ok=True)

os.makedirs(GRAPH_DIR, exist_ok=True)



                                   

sns.set_theme(style="whitegrid", context="paper")

plt.rcParams.update({

    'font.size': 12,

    'axes.titlesize': 14,

    'axes.labelsize': 12,

    'xtick.labelsize': 10,

    'ytick.labelsize': 10,

    'figure.autolayout': True,

    'figure.dpi': 300

})



              

sit1 = pd.read_csv(r"C:\IP\fruit_surface_detection\outputs\situation1.csv")

sit2 = pd.read_csv(r"C:\IP\fruit_surface_detection\outputs\situation2.csv")

sit3 = pd.read_csv(r"C:\IP\fruit_surface_detection\outputs\situation3.csv")

hsv_df = pd.read_csv(r"C:\IP\fruit_ripeness\outputs\ripeness_assessment_hsv.csv")

lab_df = pd.read_csv(r"C:\IP\fruit_ripeness\outputs\ripeness_assessment_lab.csv")



                                            

                                 

                                            

summary = []



       

for alg in sit1['algorithm'].unique():

    overlap = sit1[sit1['algorithm']==alg]['mean_prediction_overlap'].mean()

    time_s = sit1[sit1['algorithm']==alg]['processing_time_s'].mean()

    selected = "Yes" if alg == "LAB K-Means" else "No"

    reason = "Highest prediction overlap" if selected == "Yes" else "Lower overlap"

    summary.append(["Surface Detection", "Situation 1", alg, "Prediction Overlap", overlap, selected, reason, "TP=0 (Bounding Box GT)"])

    summary.append(["Surface Detection", "Situation 1", alg, "Processing Time (s)", time_s, selected, reason, "TP=0 (Bounding Box GT)"])



       

for alg in sit2['algorithm'].unique():

    overlap = sit2[sit2['algorithm']==alg]['mean_prediction_overlap'].mean()

    time_s = sit2[sit2['algorithm']==alg]['processing_time_s'].mean()

    selected = "Yes" if alg == "Morphological Watershed" else "No"

    reason = "Much higher overlap + fewer fragmented predictions" if selected == "Yes" else "Edge fragmentations"

    summary.append(["Surface Detection", "Situation 2", alg, "Prediction Overlap", overlap, selected, reason, "TP=0 (Bounding Box GT)"])

    summary.append(["Surface Detection", "Situation 2", alg, "Processing Time (s)", time_s, selected, reason, "TP=0 (Bounding Box GT)"])



       

for alg in sit3['algorithm'].unique():

    circ = sit3[sit3['algorithm']==alg]['circularity'].mean()

    time_s = sit3[sit3['algorithm']==alg]['processing_time_s'].mean()

    summary.append(["Surface Detection", "Situation 3", alg, "Circularity", circ, "Yes", "Better shape representation", "No pixel-level GT"])

    summary.append(["Surface Detection", "Situation 3", alg, "Processing Time (s)", time_s, "Yes", "Better shape representation", "No pixel-level GT"])



          

hsv_time = hsv_df['processing_time_s'].mean()

lab_time = lab_df['processing_time_s'].mean()

summary.append(["Fruit Ripeness", "HSV vs LAB", "HSV", "Processing Time (s)", hsv_time, "No Final Selection", "No GT ripeness labels available", "Unsupervised"])

summary.append(["Fruit Ripeness", "HSV vs LAB", "LAB", "Processing Time (s)", lab_time, "No Final Selection", "No GT ripeness labels available", "Unsupervised"])



df_summary = pd.DataFrame(summary, columns=["Module", "Situation", "Algorithm", "Metric", "Result", "Selected", "Reason", "Limitation"])

df_summary.to_csv(os.path.join(OUT_CSV_DIR, "results_summary.csv"), index=False)





                                            

                           

                                            



                                 

fig, ax1 = plt.subplots(figsize=(8, 5))

sns.barplot(data=sit1, x='algorithm', y='mean_prediction_overlap', ax=ax1, palette="Blues_d")

ax1.set_title("Situation 1: Prediction Overlap by Algorithm")

ax1.set_ylabel("Mean Prediction Overlap")

ax1.set_xlabel("Algorithm")

plt.savefig(os.path.join(GRAPH_DIR, "1_surface_sit1_comparison.png"))

plt.close()



                                 

fig, ax1 = plt.subplots(figsize=(8, 5))

sns.barplot(data=sit2, x='algorithm', y='mean_prediction_overlap', ax=ax1, palette="Greens_d")

ax1.set_title("Situation 2: Prediction Overlap by Algorithm")

ax1.set_ylabel("Mean Prediction Overlap")

ax1.set_xlabel("Algorithm")

plt.savefig(os.path.join(GRAPH_DIR, "2_surface_sit2_comparison.png"))

plt.close()



                                                  

fig, ax1 = plt.subplots(figsize=(8, 5))

melted_sit3 = sit3.melt(id_vars=['algorithm'], value_vars=['circularity', 'solidity'], var_name='Shape Metric', value_name='Value')

sns.boxplot(data=melted_sit3, x='Shape Metric', y='Value', hue='algorithm', palette="Oranges_d", ax=ax1)

ax1.set_title("Situation 3: Shape Descriptors Distribution")

ax1.set_ylabel("Metric Value")

plt.savefig(os.path.join(GRAPH_DIR, "3_surface_sit3_shape.png"))

plt.close()



                                                          

hsv_counts = hsv_df['predicted_ripeness'].value_counts().rename("HSV")

lab_counts = lab_df['predicted_ripeness'].value_counts().rename("LAB")

ripeness_comp = pd.concat([hsv_counts, lab_counts], axis=1).fillna(0).reset_index()

ripeness_comp = ripeness_comp.rename(columns={'predicted_ripeness': 'Ripeness', 'index': 'Ripeness'})

melted_ripeness = ripeness_comp.melt(id_vars=['Ripeness'], value_vars=['HSV', 'LAB'], var_name='Algorithm', value_name='Count')



fig, ax1 = plt.subplots(figsize=(8, 5))

sns.barplot(data=melted_ripeness, x='Ripeness', y='Count', hue='Algorithm', palette="Set2", ax=ax1)

ax1.set_title("Fruit Ripeness: Prediction Distribution (HSV vs LAB)")

ax1.set_ylabel("Number of Images")

plt.savefig(os.path.join(GRAPH_DIR, "4_ripeness_prediction_comparison.png"))

plt.close()



                                                           

time_data = []

for alg in sit1['algorithm'].unique():

    time_data.append({"Algorithm": f"S1: {alg}", "Time (s)": sit1[sit1['algorithm']==alg]['processing_time_s'].mean(), "Module": "Surface"})

for alg in sit2['algorithm'].unique():

    time_data.append({"Algorithm": f"S2: {alg}", "Time (s)": sit2[sit2['algorithm']==alg]['processing_time_s'].mean(), "Module": "Surface"})

for alg in sit3['algorithm'].unique():

    time_data.append({"Algorithm": f"S3: {alg}", "Time (s)": sit3[sit3['algorithm']==alg]['processing_time_s'].mean(), "Module": "Surface"})

time_data.append({"Algorithm": "Ripeness: HSV", "Time (s)": hsv_time, "Module": "Ripeness"})

time_data.append({"Algorithm": "Ripeness: LAB", "Time (s)": lab_time, "Module": "Ripeness"})



df_time = pd.DataFrame(time_data)

fig, ax1 = plt.subplots(figsize=(10, 6))

sns.barplot(data=df_time, y='Algorithm', x='Time (s)', hue='Module', dodge=False, palette="mako", ax=ax1)

ax1.set_title("Average Processing Time across All Algorithms")

ax1.set_xlabel("Processing Time (seconds)")

plt.savefig(os.path.join(GRAPH_DIR, "5_processing_time_all.png"))

plt.close()



print("Thesis Results Generation Complete!")

print(f"CSV saved at: {os.path.join(OUT_CSV_DIR, 'results_summary.csv')}")

print(f"Graphs saved at: {GRAPH_DIR}")

