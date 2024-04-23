# Test the model
import torch
import sklearn.metrics
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
def test(test_loader, device, model, true_labels):
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
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 score: {f1_score}")
    
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    
    text_spacer = len(true_labels) + 1
    
    plt.text(text_spacer,1,f"Precision:{round(precision,4)}")
    plt.text(text_spacer,2,f"Recall: {round(recall,4)}")
    plt.text(text_spacer,3,f"F1 score: {round(f1_score,4)}")
    
    plt.ylabel("Ground Truth", fontsize = 15)
    plt.xlabel("Predicted", fontsize = 15)
    
    plt.savefig("confusionMatrix.png")
    plt.show()
