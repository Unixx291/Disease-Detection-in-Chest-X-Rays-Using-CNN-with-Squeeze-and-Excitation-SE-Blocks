# Disease-Detection-in-Chest-X-Rays-Using-CNN-with-Squeeze-and-Excitation-SE-Blocks
CNN model enhanced with Squeeze-and-Excitation blocks for disease detection in chest X-rays. Trained on the NIH dataset using PyTorch with data augmentation and 5-fold cross-validation. Achieved 66% test accuracy, improving feature focus and diagnostic reliability.

**Description:**
This project applies deep learning to medical imaging, focusing on early disease detection in chest X-rays through a Convolutional Neural Network (CNN) enhanced with a Squeeze-and-Excitation (SE) block. The SE mechanism allows the network to dynamically emphasize the most relevant image features, improving diagnostic performance and interpretability.

The model was developed using PyTorch and trained on the NIH Chest X-Ray Dataset (≈5,600 labeled images). The data pipeline included preprocessing steps such as grayscale conversion, Gaussian blur, normalization, and 5-fold cross-validation to ensure robust evaluation.

**Technical Details:**
Architecture: Custom CNN integrated with SE block for adaptive channel weighting

Learning Rate: 1×10⁻⁴ with StepLR scheduler (decay factor 0.7 every 2 epochs)

Optimizer: Adam with weight decay regularization

Batch Size: 32, over 30 epochs

Evaluation Metrics: Training/validation accuracy, validation loss, and testing accuracy on 1,000 unseen images

**Results:**
Simple CNN: 50% test accuracy

Improved CNN: 57–62% test accuracy

CNN + SE Block: 66% test accuracy

The SE-enhanced model showed notable gains in classification accuracy and robustness, proving effective in highlighting disease-specific features across variable image quality. Despite limited computational resources (CPU-only training), the model demonstrated strong generalization and potential for clinical applications.

**Future Work:**
Further improvements could include larger datasets, GPU-based training, dropout regularization, and fine-tuning with pretrained networks like ResNet50 or DenseNet121 to enhance precision and recall.

**Key Skills & Tools:**
Python, PyTorch, Deep Learning, Convolutional Neural Networks, Medical Image Analysis, Data Augmentation, Cross-Validation, Model Evaluation, Scientific Reporting.

This repository demonstrates practical experience in AI for healthcare—covering data preprocessing, model design, experimental evaluation, and result interpretation—making it ideal for employers seeking candidates skilled in applied machine learning and computer vision.
