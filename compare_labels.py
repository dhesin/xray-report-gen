import pandas as pd
import os
import numpy as np
from sklearn.metrics import classification_report

#os.system("docker run -v $(pwd):/data chexpert-labeler:latest   python label.py --reports_path /data/preds_only_generated.csv --output_path /data/labeled_preds.csv --verbose")


df_1 = pd.read_csv("./reports_preds_generated.csv").reset_index()
df_2 = pd.read_csv("./labeled_preds.csv").reset_index()

assert(len(df_1) == len(df_2))


df_merged = df_1.merge(df_2, how='inner', left_index=True, right_index=True)
df_merged.to_csv("./merged_preds_labels.csv")

column_names = ['No Finding_y','Enlarged Cardiomediastinum_y','Cardiomegaly_y','Lung Lesion_y', \
                              'Lung Opacity_y','Edema_y','Consolidation_y','Pneumonia_y','Atelectasis_y',\
                              'Pneumothorax_y','Pleural Effusion_y','Pleural Other_y','Fracture_y','Support Devices_y']

# Labels from predicted report
labels_from_generated_report = df_merged[['No Finding_y','Enlarged Cardiomediastinum_y','Cardiomegaly_y','Lung Lesion_y', \
                              'Lung Opacity_y','Edema_y','Consolidation_y','Pneumonia_y','Atelectasis_y',\
                              'Pneumothorax_y','Pleural Effusion_y','Pleural Other_y','Fracture_y','Support Devices_y']].fillna(2).to_numpy().astype(int)


# Ground truths
labels_from_original_report = df_merged[['No Finding_x','Enlarged Cardiomediastinum_x','Cardiomegaly_x','Lung Lesion_x', \
                              'Lung Opacity_x','Edema_x','Consolidation_x','Pneumonia_x','Atelectasis_x',\
                              'Pneumothorax_x','Pleural Effusion_x','Pleural Other_x','Fracture_x','Support Devices_x']].to_numpy()
# subtract the 1  which is added to labels to shift range 0-3 in dataset.py
labels_from_original_report = labels_from_original_report - 1


compare = np.where(labels_from_generated_report == labels_from_original_report, 1, 0)
#print(compare)

accuracy = float(compare.sum())/float(compare.size)
print("accuracy of the labels:", accuracy)

print(labels_from_generated_report.shape)
f1_scores = []
precisions = []
recalls = []
for label in range(labels_from_generated_report.shape[1]):
    print(column_names[label])
    cls_dict  = classification_report(labels_from_original_report[:,label], labels_from_generated_report[:,label], output_dict=True)
    print(cls_dict)
    #cls_report  = classification_report(labels_from_original_report[:,label], labels_from_generated_report[:,label])
    f1_score = cls_dict['macro avg']['f1-score']
    precision = cls_dict['macro avg']['precision']
    recall = cls_dict['macro avg']['recall']
    f1_scores.append(f1_score)
    precisions.append(precision)
    recalls.append(recall)
    #print(column_names[label], "/ F1 score:", f1_score, " Recall:", recall, " Precision:", precision)
    #print(cls_report)
print("avg f1 score:", sum(f1_scores)/len(f1_scores))
print("avg precision:", sum(precisions)/len(precisions))
print("avg recalls:", sum(recalls)/len(recalls))

