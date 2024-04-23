# Test the model
import torch
import sklearn.metrics
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
def test(test_loader, device, model, true_labels, graph_title=None):
    y_pred = []
    y_true = []
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for hands, labels in test_loader:
            hands = hands.to(device)
            labels = labels.to(device)
            outputs = model(hands)
            output = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
            y_pred.extend(output)  # Save Prediction

            labels = labels.data.cpu().numpy()
            y_true.extend(labels)  # Save Truth

    # Build confusion matrix
    cf_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(
        cf_matrix / np.sum(cf_matrix, axis=1)[:, None],
        index=[i for i in true_labels],
        columns=[i for i in true_labels],
    )

    precision = sklearn.metrics.precision_score(y_true, y_pred, average="weighted")
    recall = sklearn.metrics.recall_score(y_true, y_pred, average="weighted")
    f1_score = sklearn.metrics.f1_score(y_true, y_pred, average="weighted")
    min_precision, min_recall, min_f1_score, support = sklearn.metrics.precision_recall_fscore_support(y_true, y_pred, average=None)
    
    
    worst_precision = np.argmin(min_precision)
    worst_recall = np.argmin(min_recall)
    worst_f1_score = np.argmin(min_f1_score)
    
    worst_precision = true_labels[worst_precision]
    worst_recall = true_labels[worst_recall]
    worst_f1_score = true_labels[worst_f1_score]
    
    min_precision = np.min(min_precision)
    min_recall = np.min(min_recall)
    min_f1_score = np.min(min_f1_score)
    
    print(f"support={support}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 score: {f1_score}")
    
    print(f"Worst Precision '{worst_precision}': {min_precision}")
    print(f"Worst Recall '{worst_recall}': {min_recall}")
    print(f"Worst F1 score '{worst_f1_score}': {min_f1_score}")
    
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True, cbar=False)
    
    text_spacer = len(true_labels) + 1
    
    plt.text(text_spacer,1,f"Overall Precision:{round(precision,4)}")
    plt.text(text_spacer,2,f"Overall Recall: {round(recall,4)}")
    plt.text(text_spacer,3,f"Overall F1 score: {round(f1_score,4)}")
    plt.text(text_spacer,4,f"Worst Precision '{worst_precision}': {round(min_precision, 4)}")
    plt.text(text_spacer,5,f"Worst Recall '{worst_recall}': {round(min_recall, 4)}")
    plt.text(text_spacer,6,f"Worst F1 score '{worst_f1_score}': {round(min_f1_score, 4)}")
    plt.text(text_spacer,7,"Samples per class:")
    i = 8
    for index, item in enumerate(support):
        plt.text(text_spacer,i,f"'{true_labels[index]}': {item}")
        i+=0.5
    
    plt.title(graph_title, fontsize = 15)
    plt.ylabel("Ground Truth", fontsize = 15)
    plt.xlabel("Predicted", fontsize = 15)
    
    plt.savefig("confusionMatrix.png", bbox_inches="tight")
    plt.show()
