import pandas as pd
import numpy as np
import torch
import os

from datasets import create_train_val_datasets
from architectures import SimpleCNN, ResidualCNN
from training import training_loop
from inference import predict
from submission_serialization import serialize

# load test dataset
test_set_dict = pd.read_pickle("project/data/test_set.pkl")

# create training and validation dataset
train_set, valid_set = create_train_val_datasets(image_dir="project/data/training")

# seed for reproducibility
torch.manual_seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"

# # create simple cnn model
# cnn_model = SimpleCNN(
#     input_channels=2,
#     hidden_channels=128,
#     output_channels=1,
#     num_hidden_layers=10
# ).to(device)

# create residual cnn model
residual_model = ResidualCNN(
    input_channels=2,
    hidden_channels=200,
    output_channels=1,
    squeeze_channels=128,
    num_blocks=5,
    kernel_size=3
).to(device)
# train network
train_losses, eval_losses, saved_model_file, run_name = training_loop(
    network=residual_model, train_data=train_set, val_data=valid_set,
    num_epochs=100, last_n_epochs=10, learning_rate=0.0001, batch_size=64,
    show_progress=True, num_logged_images=6, device=device, models_path="project/saved_models/"
)

# predictions on the test set
best_network = torch.load(saved_model_file)
predictions = predict(network=best_network, test_set_dict=test_set_dict, batch_size=32, device=device)

# transform predictions to the correct shape for the challenge server
serialize(submission=predictions, path_or_filehandle=os.path.join("project/predictions", f"{run_name}.data"))
