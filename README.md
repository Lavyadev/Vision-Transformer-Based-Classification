# Vision Transformer (ViT) Based Classification

This repository contains code for classifying synthetic datasets using a Vision Transformer (ViT) model. The task involves classifying images into one of two datasets: `Run355456_Dataset.npy` and `Run357479_Dataset.npy`.

## Model Architecture
The model is based on a Vision Transformer (ViT) architecture, specifically the `vit_b_16` variant pretrained on ImageNet. The model was modified for binary classification by replacing the final classification head with a linear layer that outputs 2 classes. Key hyperparameters include:
- **Learning Rate**: `5e-5`
- **Batch Size**: `64`
- **Optimizer**: AdamW with weight decay (`1e-4`)
- **Loss Function**: CrossEntropyLoss with label smoothing (`0.1`)
- **Epochs**: `30`
- **Learning Rate Scheduler**: ReduceLROnPlateau with a factor of `0.5` and patience of `3`

## Data Preprocessing
The datasets consist of grayscale images with shape `(64, 72)`. The following preprocessing steps were applied:
1. **Resizing**: Images were resized to `(224, 224)` to match the input size of the ViT model.
2. **Augmentation**: Random horizontal flipping, random rotation (15 degrees), and AutoAugment were applied to improve generalization.
3. **Normalization**: Images were normalized using the dataset mean and standard deviation.
4. **Grayscale to RGB**: Single-channel grayscale images were converted to 3-channel RGB images by repeating the grayscale values across all channels.

## Evaluation Metrics
The model's performance was evaluated using the following metrics:
- **Accuracy**: `0.7380`
- **AUC**: The Area Under the Curve (AUC) was calculated as `0.8112`, indicating good model performance.
- **ROC Curve**: The Receiver Operating Characteristic (ROC) curve was plotted to visualize the trade-off between true positive rate (TPR) and false positive rate (FPR).
  ![ROC Curve](roc_curve.png)

## Results
- **Accuracy**: `0.7380`
- **AUC**: `0.8112`

