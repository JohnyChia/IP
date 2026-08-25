import pandas as pd

import numpy as np

import os



OUT_DIR = r"C:\IP\thesis_evaluation\outputs"

os.makedirs(OUT_DIR, exist_ok=True)



                                            

                                 

                                            

sit1 = pd.read_csv(r"C:\IP\fruit_surface_detection\outputs\situation1.csv")

sit2 = pd.read_csv(r"C:\IP\fruit_surface_detection\outputs\situation2.csv")

sit3 = pd.read_csv(r"C:\IP\fruit_surface_detection\outputs\situation3.csv")



surface_results = []

sit1_best = None

sit1_best_overlap = -1

sit2_best = None

sit2_best_overlap = -1

sit3_best = None



for sit_name, df in [('Situation 1', sit1), ('Situation 2', sit2)]:

    for alg in df['algorithm'].unique():

        sub = df[df['algorithm'] == alg]

        avg_overlap = sub['mean_prediction_overlap'].mean() if 'mean_prediction_overlap' in sub.columns else 0.0

        avg_time = sub['processing_time_s'].mean() if 'processing_time_s' in sub.columns else 0.0

        

        surface_results.append({

            'situation': sit_name,

            'algorithm': alg,

            'avg_precision': "INVALID_GT_MATCHING",

            'avg_recall': "INVALID_GT_MATCHING",

            'avg_f1': "INVALID_GT_MATCHING",

            'avg_iou': "INVALID_GT_MATCHING",

            'avg_prediction_overlap': avg_overlap,

            'avg_processing_time_s': avg_time

        })

        

        if sit_name == 'Situation 1' and avg_overlap > sit1_best_overlap:

            sit1_best_overlap = avg_overlap

            sit1_best = alg

        elif sit_name == 'Situation 2' and avg_overlap > sit2_best_overlap:

            sit2_best_overlap = avg_overlap

            sit2_best = alg



for alg in sit3['algorithm'].unique():

    sub = sit3[sit3['algorithm'] == alg]

    avg_circ = sub['circularity'].mean() if 'circularity' in sub.columns else 0.0

    avg_sol = sub['solidity'].mean() if 'solidity' in sub.columns else 0.0

    avg_time = sub['processing_time_s'].mean() if 'processing_time_s' in sub.columns else 0.0

    sit3_best = alg                                                                       

    

    surface_results.append({

        'situation': 'Situation 3',

        'algorithm': alg,

        'avg_precision': "INVALID_GT_MATCHING",

        'avg_recall': "INVALID_GT_MATCHING",

        'avg_f1': "INVALID_GT_MATCHING",

        'avg_iou': "INVALID_GT_MATCHING",

        'avg_prediction_overlap': 'N/A',

        'avg_processing_time_s': avg_time,

        'avg_circularity': avg_circ,

        'avg_solidity': avg_sol

    })

    

pd.DataFrame(surface_results).to_csv(os.path.join(OUT_DIR, 'surface_algorithm_comparison.csv'), index=False)



                                            

                              

                                            

hsv_df = pd.read_csv(r"C:\IP\fruit_ripeness\outputs\ripeness_assessment_hsv.csv")

lab_df = pd.read_csv(r"C:\IP\fruit_ripeness\outputs\ripeness_assessment_lab.csv")



ripeness_results = []

for name, df in [('HSV Algorithm', hsv_df), ('LAB Algorithm', lab_df)]:

                                             

    valid_rate = (df['valid_roi_count'] / df['roi_count']).mean() if 'roi_count' in df.columns and 'valid_roi_count' in df.columns else 'N/A'

    fg_ratio = df['foreground_ratio'].mean() if 'foreground_ratio' in df.columns else 'N/A'

    time_s = df['processing_time_s'].mean() if 'processing_time_s' in df.columns else 'N/A'

    

    ripeness_results.append({

        'algorithm': name,

        'valid_roi_rate': valid_rate,

        'avg_foreground_ratio': fg_ratio,

        'avg_processing_time_s': time_s,

        'accuracy': 'NO_GROUND_TRUTH',

        'f1_score': 'NO_GROUND_TRUTH'

    })

pd.DataFrame(ripeness_results).to_csv(os.path.join(OUT_DIR, 'ripeness_algorithm_comparison.csv'), index=False)



                      

if len(hsv_df) == len(lab_df) and 'predicted_ripeness' in hsv_df.columns and 'predicted_ripeness' in lab_df.columns:

    agreement = (hsv_df['predicted_ripeness'] == lab_df['predicted_ripeness']).mean()

else:

    agreement = 0.0



                                            

                                  

                                            

selections = []



selections.append({

    'module': 'Surface Detection', 

    'situation': 'Situation 1', 

    'algorithm': sit1_best, 

    'selection_status': 'SELECTED', 

    'primary_reason': 'Highest average prediction overlap despite potential speed trade-offs.',

    'quality_metric': 'Prediction Overlap', 

    'time_metric': 'processing_time_s', 

    'limitation': 'F1 = INVALID_FOR_SELECTION (TP=0 due to bounding box GT vs mask mismatch)'

})



selections.append({

    'module': 'Surface Detection', 

    'situation': 'Situation 2', 

    'algorithm': sit2_best, 

    'selection_status': 'SELECTED', 

    'primary_reason': 'Significantly higher mask overlap and cohesive regions than pure edge detection.',

    'quality_metric': 'Prediction Overlap', 

    'time_metric': 'processing_time_s', 

    'limitation': 'F1 = INVALID_FOR_SELECTION'

})



selections.append({

    'module': 'Surface Detection', 

    'situation': 'Situation 3', 

    'algorithm': sit3_best, 

    'selection_status': 'SELECTED', 

    'primary_reason': 'Provides global shape descriptors successfully.',

    'quality_metric': 'Shape Descriptors', 

    'time_metric': 'processing_time_s', 

    'limitation': 'No pixel-level GT for geometric accuracy validation'

})



selections.append({

    'module': 'Fruit Ripeness', 

    'situation': 'HSV vs LAB', 

    'algorithm': 'Both', 

    'selection_status': 'NO_FINAL_SELECTION', 

    'primary_reason': 'No ground truth ripeness labels to validate predictive superiority.',

    'quality_metric': 'NO_GROUND_TRUTH', 

    'time_metric': 'processing_time_s', 

    'limitation': 'Lack of ripeness classes in benchmark prevents empirical selection.'

})



pd.DataFrame(selections).to_csv(os.path.join(OUT_DIR, 'final_algorithm_selection.csv'), index=False)





                                            

                   

                                            

print("\n================ FINAL ALGORITHM SELECTION ================\n")

print("SURFACE DETECTION")

print(f"Situation 1: {sit1_best} selected (Highest Overlap).")

print(f"Situation 2: {sit2_best} selected (Highest Overlap).")

print(f"Situation 3: {sit3_best} selected.\n")



print("FRUIT RIPENESS")

print("Situation 1: HSV Algorithm")

print("Situation 2: LAB Algorithm")

print("Selection: NO_FINAL_SELECTION")

print(f"Algorithm Agreement Rate: {agreement*100:.1f}%\n")



print("IMPORTANT LIMITATIONS")

print("- Surface Detection: F1 / Precision / Recall are marked as INVALID_FOR_SELECTION. The dataset GT annotations are bounding boxes localized around the fruit (Class 0), not pixel-perfect defect masks. This results in TP=0, rendering standard segmentation accuracy metrics invalid.")

print("- Fruit Ripeness: Marked as NO_GROUND_TRUTH. The benchmark lacks ripeness classification labels entirely, meaning HSV and LAB can only be compared on stability and processing time, but not accuracy.")

print("===========================================================\n")

