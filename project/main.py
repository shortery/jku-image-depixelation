import pandas as pd
import torch

from datasets import create_train_val_datasets
from architectures import SimpleCNN
from training import training_loop

# load test dataset
test_set_dict = pd.read_pickle("project/data/test_set.pkl")

# create training and validation dataset
train_set, valid_set = create_train_val_datasets(image_dir="project/data/training")

# seed for reproducibility
torch.manual_seed(0)

# create simple cnn model
cnn_model = SimpleCNN(
    input_channels=2,
    hidden_channels=32,
    output_channels=1,
    num_hidden_layers=5
)

train_losses, eval_losses = training_loop(
    cnn_model, train_set, valid_set, num_epochs=7, last_n_epochs=5, learning_rate=0.0001, show_progress=True, num_logged_images=6
)