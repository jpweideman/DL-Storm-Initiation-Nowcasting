# Model Architectures

This folder contains modular definitions of all deep learning models used for radar nowcasting.

## Available Models

- **conv_lstm.py** — Convolutional LSTM model.
- **traj_gru.py** — Trajectory GRU model.
- **traj_gru_enc_dec.py** — Symmetric TrajGRU encoder-decoder model (no U-Net, no skip connections).
- **cnn_3d.py** — 3D Convolutional Neural Network.
- **unet_3d_cnn.py** — U-Net style 3D CNN.
- **unet_conv_lstm.py** — U-Net with ConvLSTM blocks.
- **unet_traj_gru.py** — U-Net with TrajGRU blocks.

## Usage
- Import models in training scripts, e.g.:
  ```python
  from src.models.unet_3d_cnn import UNet3DCNN
  ```

See [src/training/README.md](../training/README.md) for training models. 