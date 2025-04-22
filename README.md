# Tuberculosis_prediction

## 🫁 Chest X-ray Classification using Deep Learning
A project involving deep learning to classify chest X-ray images for early detection of possible chest illnesses. The model is constructed through Convolutional Neural Networks (CNNs) and trained with labeled medical image data.


🧠 Project Objective
The objective of this project is to create a deep learning model that can identify anomalies in chest X-ray images accurately. With the increasing requirement for automated medical diagnosis, particularly in under-developed regions, this project focuses on delivering a core tool to identify diseases such as pneumonia, tuberculosis, or other thoracic disorders.


📁 Dataset
The dataset utilized includes chest X-ray images, classified into various classes depending on medical diagnosis.


Data is preprocessed through resizing, normalization, and data augmentation methods to enhance generalization.

🛠️ Project Structure

├── README.md              <- You're here!
├── data/                  <- RAW and PROCESSED data directory
├── notebooks/
│   └── cheast_model.ipynb <- Model building and testing
├── src/                   <- Source folder
│   ├── data_loader.py     <- Data loading and preparation
│   ├── model.py           <- Model design
│   ├── train.py           <- Script de formação
│   └── evaluate.py        <- Metrics de avaliação do modelo
├── app/                   <- Opcional: Streamlit ou Flask app de demo
├── requirements.txt       <- Python dependencies
└──.gitignore             <- Files to ignore in version control



🧪 Model Architecture
The model is a CNN constructed with TensorFlow/Keras. Major features:

Convolutional layers with ReLU activation
MaxPooling layers
Dropout regularization
Fully connected layers for final classification

📊 Evaluation Metrics
Accuracy
Precision, Recall, F1-score
Confusion Matrix
ROC-AUC (if multi-label or binary)
Loss vs. Accuracy plots at training

🚀 Deployment
To deploy the model, a basic web app is deployed using Streamlit (or Flask) where the user can upload an X-ray and receive a predicted label.

Try it out locally:
streamlit run app/app.py

⚠️ Ethical Considerations
This model is not for real clinical use without medical corroboration.
Accuracy can vary depending on data distribution, quality, and biases.
Always get a licensed medical practitioner's opinion for true diagnosis.
