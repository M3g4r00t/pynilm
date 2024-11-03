from sklearn.metrics import *

# Custom metric (f1_macro)
def f1_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

# Custom metric (precision_score)
def precision_macro(y_true, y_pred):
    return precision_score(y_true, y_pred, average='macro')

# Custom metric (recall_macro)
def recall_macro(y_true, y_pred):
    return recall_score(y_true, y_pred, average='macro')