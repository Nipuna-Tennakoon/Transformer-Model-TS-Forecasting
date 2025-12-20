# Time Series Forecasting for Biomass Heater IoT Data

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-orange)](https://pytorch.org/)

## 📜 Overview

This project implements a **Transformer-based neural network** for forecasting **Serpentine Coil Inlet Temperature** in biomass heater systems using IoT sensor data. The model leverages the self-attention mechanism of Transformers to capture temporal dependencies and patterns in temperature time series data, enabling accurate one-step-ahead predictions.

## 🔥 Key Features

- **Transformer Architecture**: Custom Transformer model specifically designed for time series temperature forecasting
- **IoT Data Processing**: Handles real-world biomass heater sensor data with proper preprocessing
- **Logarithmic Normalization**: Uses log returns for data normalization to improve model stability
- **Attention Mechanism**: Exploits self-attention to capture both short-term and long-term temperature dependencies
- **Visualization**: Comprehensive plotting of actual vs predicted temperature sequences
- **Model Validation**: Validates on multiple temperature features from the same IoT system

## 🎯 Problem Statement

The goal is to forecast the **Average T.F. Heater Serpentine Coil Inlet Temperature** using historical temperature data from a biomass heater IoT system. The model uses 10 historical timesteps to predict the next temperature value, which is crucial for:
- Predictive maintenance
- Energy optimization
- System monitoring and control
- Anomaly detection

## 🛠️ Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/Time-Series-Forecasting.git
cd Time-Series-Forecasting
pip install -r requirements.txt
```

### Dependencies

- `torch>=1.8.0` - PyTorch for deep learning
- `numpy>=1.19.5` - Numerical computations
- `pandas>=1.2.3` - Data manipulation
- `matplotlib>=3.3.4` - Visualization

## 📂 Repository Structure

```plaintext
.
├── data/                                          # Folder to store IoT sensor datasets
│   └── final_df_002.csv                          # Biomass heater IoT data
├── model/                                         # Folder containing saved models
│   └── transformer_model.pth                     # Trained transformer model
├── src/                                           # Source code
│   ├── main.ipynb                                # Jupyter notebook for training and forecasting
│   └── main.py                                   # Python script version
├── requirements.txt                               # Python dependencies
└── README.md                                      # Project documentation
```

## 📊 Dataset

The project uses IoT sensor data from a biomass heater system (`final_df_002.csv`). The dataset contains multiple temperature measurements including:

- **Average T.F. Heater Serpentine Coil Inlet Temperature** (primary forecast target)
- Average T.F. Heater Outlet Temperature
- T.F. Heater Serpentine Coil Outlet Temperature

### Data Preprocessing

The model applies logarithmic normalization to the temperature data:
1. Computes log returns: `log_return = diff(log(temperature))`
2. Applies cumulative sum for normalization
3. Creates sliding window sequences (10 timesteps → 1 prediction)

## 🚀 Usage

### Training the Model

Open the Jupyter notebook and run the cells:

```bash
jupyter notebook src/main.ipynb
```

The training process includes:
1. **Data Loading**: Loads biomass heater IoT data
2. **Preprocessing**: Applies logarithmic normalization
3. **Model Training**: Trains Transformer for 100 epochs
4. **Validation**: Evaluates on validation set (20% of data)
5. **Model Saving**: Saves trained model to `model/transformer_model.pth`

### Model Configuration

- **Input Window**: 10 timesteps
- **Output Window**: 1 timestep (one-step-ahead forecast)
- **Batch Size**: 64
- **Learning Rate**: 0.0005 (with StepLR scheduler, gamma=0.95)
- **Epochs**: 100
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: AdamW
- **Train/Validation Split**: 80/20

## 🧠 Model Architecture

The Transformer model consists of:

### Components

1. **Positional Encoding Layer**
   - Adds positional information to input sequences
   - Uses sinusoidal encoding for temporal position

2. **Transformer Encoder**
   - **Feature Size**: 250 dimensions
   - **Number of Layers**: 1
   - **Attention Heads**: 10
   - **Dropout**: 0.1
   - **Self-Attention Mechanism**: Captures temporal relationships

3. **Decoder**
   - Linear layer mapping encoder output to single temperature prediction

### Architecture Flow

```
Input Sequence (10 timesteps) 
  → Positional Encoding 
  → Transformer Encoder (with self-attention)
  → Linear Decoder 
  → Predicted Temperature (1 timestep)
```

## 📈 Training Process

The model training includes:

1. **Data Augmentation**: Training data is scaled by a factor of 2 to improve generalization
2. **Gradient Clipping**: Maximum gradient norm of 0.7 to prevent exploding gradients
3. **Learning Rate Scheduling**: StepLR scheduler reduces learning rate by 5% each epoch
4. **GPU Support**: Automatically uses CUDA if available

### Training Output

The training loop provides:
- Batch-level loss updates
- Epoch completion time
- Validation loss after training

## 🔍 Model Validation

The trained model is validated on additional temperature features:

1. **Average T.F. Heater Outlet Temperature**
   - Test loss: ~0.00008

2. **T.F. Heater Serpentine Coil Outlet Temperature**
   - Test loss: ~0.00007

This demonstrates the model's ability to generalize across related temperature measurements in the same IoT system.

## 💾 Model Persistence

The trained model is saved as:
```
model/transformer_model.pth
```

To load and use the saved model:
```python
model = transformer()
model.load_state_dict(torch.load("model/transformer_model.pth"))
model.to(device)
```

## 🔬 Key Technical Details

### Data Normalization
- Uses logarithmic returns to handle non-stationary temperature data
- Cumulative sum applied to create normalized sequences
- Training data augmented with 2x scaling

### Sequence Creation
- Sliding window approach: 10 consecutive timesteps → 1 prediction
- Sequences prepared as PyTorch tensors on GPU

### Forecasting Functions
- `forecast_seq()`: Forecasts entire validation sequence
- `model_forecast()`: Single timestep forecast from window sequence

## 🤝 Contributing

Contributions are welcome! If you have ideas, suggestions, or bug reports:

1. Fork the project
2. Create your feature branch (`git checkout -b feature/NewFeature`)
3. Commit your changes (`git commit -m 'Add some NewFeature'`)
4. Push to the branch (`git push origin feature/NewFeature`)
5. Open a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙌 Acknowledgements

Special thanks to the open-source community and developers who contribute to advancing machine learning, time series forecasting, and IoT data analysis.

## 📧 Contact

For questions or issues, please open an issue on GitHub.

