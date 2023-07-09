import torch
import numpy as np
from tqdm import tqdm
import wandb

def training_loop(
    network: torch.nn.Module,
    train_data: torch.utils.data.Dataset,
    eval_data: torch.utils.data.Dataset,
    num_epochs: int,
    last_n_epochs: int,
    show_progress: bool = False
    ) -> tuple[list, list]:
    """Training loop with early stopping criterion included"""

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=32)
    valid_dataloader  =torch.utils.data.DataLoader(eval_data, batch_size=32)
    
    optimizer = torch.optim.Adam(network.parameters(), lr=0.005)
    loss_function = torch.nn.MSELoss()

    train_losses = []
    eval_losses = []

    # create wandb run
    wandb.init(project="jku-image-depixelation")

    for epoch in tqdm(range(num_epochs), disable=not show_progress):
        train_epoch_losses = []
        valid_epoch_losses = []

        # training the network 
        network.train()
        for concat_pixelated_known, original_image in train_dataloader:
            output = network(concat_pixelated_known)  # Get model output (forward pass)

            known_array = concat_pixelated_known[:, 1:2, :, :]
            output_sliced = output[known_array == False]
            target_tensor = original_image[known_array == False]

            loss = loss_function(output_sliced, target_tensor)  # Compute loss
            wandb.log({"train-loss": loss})

            loss.backward()  # Compute gradients (backward pass)
            optimizer.step()  # Perform gradient descent update step
            optimizer.zero_grad()  # Reset gradients
            train_epoch_losses.append(loss.item()) # training loss on the minibatch
        averaged_train_loss = np.average(train_epoch_losses) # trainig losses averaged over all minibatch losses
        train_losses.append(averaged_train_loss) 
        wandb.log({"avg-train-loss": averaged_train_loss})

        # evaluating the network
        network.eval()
        with torch.no_grad():
            for concat_pixelated_known, original_image in valid_dataloader:
                output = network(concat_pixelated_known)

                known_array = concat_pixelated_known[:, 1:2, :, :]
                output_sliced = output[known_array == False]
                target_tensor = original_image[known_array == False]

                loss = loss_function(output_sliced, target_tensor)
                valid_epoch_losses.append(loss.item())
        averaged_val_loss = np.average(valid_epoch_losses)
        eval_losses.append(averaged_val_loss)
        wandb.log({"avg-valid-loss": averaged_val_loss})

        # early stopping criterion
        if len(eval_losses) > last_n_epochs:
            if min(eval_losses[:-last_n_epochs]) <= min(eval_losses[-last_n_epochs:]):
                return train_losses, eval_losses
    
    return train_losses, eval_losses

