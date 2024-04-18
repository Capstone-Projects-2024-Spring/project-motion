# Test the model
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
def test(test_loader, device, lstm, true_labels):
    y_pred = []
    y_true = []
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for hands, labels in test_loader:
            hands = hands.to(device)
            labels = labels.to(device)
            outputs = lstm(hands)
            output = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
            y_pred.extend(output)  # Save Prediction

            labels = labels.data.cpu().numpy()
            y_true.extend(labels)  # Save Truth

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(
        cf_matrix / np.sum(cf_matrix, axis=1)[:, None],
        index=[i for i in true_labels],
        columns=[i for i in true_labels],
    )


    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)

    plt.savefig("confusionMatrix.png")
    plt.show()