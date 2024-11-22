# Chess-pieces Recognition System  

This repository implements a **Chess-pieces Recognition System** that classifies and localizes chess pieces on a chessboard using advanced deep learning models. The system is designed to provide high accuracy and efficiency, making it suitable for real-time applications in chess-related tasks.  

---

## **Features**
- **State-of-the-art models:** Utilizes pre-trained models like **MobileNetV2**, **EfficientNet-B7**, and others for feature extraction and classification.  
- **Customizable architecture:** Models can be fine-tuned for specific datasets.  
- **High accuracy:** Inspired and improved based on findings from published research papers.  
- **Ease of use:** Supports easy integration for inference and evaluation.  

---

## **Requirements**
Before you begin, ensure you have the following installed:  
- Python >= 3.8  
- PyTorch >= 1.11  
- torchvision >= 0.12  
- Other dependencies (install via `requirements.txt`):  
  ```bash
  pip install -r requirements.txt
  ```

---

## **Datasets**
- **Dataset format:** Chessboard images annotated with labels and bounding boxes for each chess piece.  
- **Source:** Custom dataset or publicly available chess datasets.  

You can preprocess and organize your dataset using the following structure:  
```plaintext
data/
  train/
    Bishop/
    King/
    ...
  val/
    Bishop/
    King/
    ...
```

---

## **Usage**  

### **1. Clone the repository**
```bash
git clone https://github.com/bach04py/Chess-pieces-Recognition-sytem.git
cd Chess-pieces-Recognition-sytem
```

### **2. Train the model**
Run the following command to start training:  
```bash
python train.py --model mobilenetv2 --epochs 30 --batch_size 64
```

### **3. Evaluate the model**
To evaluate the model on the validation dataset:  
```bash
python evaluate.py --model efficientnet_b7 --checkpoint path/to/checkpoint.pth
```

### **4. Perform inference**
To recognize chess pieces from an input image:  
```bash
python inference.py --image path/to/image.jpg --model path/to/checkpoint.pth
```

---

## **Models Used**
1. **MobileNetV2**: Lightweight and efficient, ideal for real-time tasks.  
2. **EfficientNet-B7**: Provides high accuracy with an optimized parameter count.  

### **Why these models?**
The models were selected based on their performance as demonstrated in leading academic papers. They balance computational efficiency and accuracy, making them suitable for chess piece recognition.  

---

## **References**  
This project draws heavily on insights from the following research papers:  
- **"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"**  
- **"MobileNetV2: Inverted Residuals and Linear Bottlenecks"**  

---

## **Contributions**  
Feel free to contribute to this repository. Submit issues, pull requests, or suggestions to improve this project.  

---

## **License**  
This repository is licensed under the MIT License. See the `LICENSE` file for details.  

---

If you find this repository helpful, please star it 🌟! 
