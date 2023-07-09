import pandas as pd
import torch

from datasets import create_train_val_datasets
from architectures import SimpleCNN
from training import training_loop

# load test dataset
test_set_dict = pd.read_pickle("project/data/test_set.pkl")

# create training and validation dataset
train_set, valid_set = create_train_val_datasets(image_dir="project/data/playground_training")

# seed for reproducibility
torch.manual_seed(0)

# create simple cnn model
cnn_model = SimpleCNN(
    input_channels=2,
    hidden_channels=24,
    output_channels=1,
    num_hidden_layers=5
)

train_losses, eval_losses = training_loop(
    cnn_model, train_set, valid_set, num_epochs=50, last_n_epochs=30, show_progress=True
)
print("", flush=True)
for epoch, (tl, el) in enumerate(zip(train_losses, eval_losses)):
    print(f"Epoch: {epoch} --- Train loss: {tl:7.2f} --- Eval loss: {el:7.2f}")

