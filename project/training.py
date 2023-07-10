import torch
import torch.utils.data
import numpy as np
from tqdm import tqdm
import wandb
import os

def training_loop(
    network: torch.nn.Module,
    train_data: torch.utils.data.Dataset,
    val_data: torch.utils.data.Dataset,
    num_epochs: int,
    last_n_epochs: int,
    learning_rate: float,
    batch_size: int,
    show_progress: bool,
    num_logged_images: int,
    device: str,
    models_path: str
    ) -> tuple[list, list]:
    """Training loop with early stopping criterion included"""

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, pin_memory=True, num_workers=4, prefetch_factor=16, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, pin_memory=True, num_workers=4, prefetch_factor=16)
    
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    loss_function = torch.nn.MSELoss()

    train_losses = []
    val_losses = []

    # create wandb run
    wandb.init(project="jku-image-depixelation")

    # save initial model as "best" model (will be overwritten later)
    best_val_loss = np.inf
    run_name = wandb.run.name
    saved_model_file = os.path.join(models_path, f"{wandb.run.name}.pt")
    torch.save(network, saved_model_file)

    for epoch in tqdm(range(num_epochs), disable=not show_progress):
        train_epoch_losses = []
        valid_epoch_losses = []

        # training the network 
        network.train()
        for concat_pixelated_known, original_image in train_dataloader:
            # move to device
            concat_pixelated_known = concat_pixelated_known.to(device)
            original_image = original_image.to(device)

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
        wandb.log({"train-avg-loss": averaged_train_loss})

        # evaluating the network
        network.eval()
        with torch.no_grad():
            for concat_pixelated_known, original_image in valid_dataloader:
                # move to device
                concat_pixelated_known = concat_pixelated_known.to(device)
                original_image = original_image.to(device)

                output = network(concat_pixelated_known)

                pixelated_array = concat_pixelated_known[:, :1, :, :]
                known_array = concat_pixelated_known[:, 1:2, :, :]
                output_sliced = output[known_array == False]
                target_tensor = original_image[known_array == False]

                loss = loss_function(output_sliced, target_tensor)
                valid_epoch_losses.append(loss.item())

        averaged_val_loss = np.average(valid_epoch_losses)
        val_losses.append(averaged_val_loss)
        wandb.log({"valid-avg-loss": averaged_val_loss})

        # save best model
        if averaged_val_loss < best_val_loss:
            best_val_loss = averaged_val_loss
            torch.save(network, saved_model_file)
        
        for i in range(num_logged_images):
            images = np.concatenate([pixelated_array[i].cpu(), output[i].cpu(), original_image[i].cpu()], axis=2)
            wandb.log({
                f"images/concatenated_img_{i}": wandb.Image(images, caption='Pixelated,    Predicted,    Original')
            })

        # early stopping criterion
        if len(val_losses) > last_n_epochs:
            if min(val_losses[:-last_n_epochs]) <= min(val_losses[-last_n_epochs:]):
                return train_losses, val_losses, saved_model_file, run_name
    
    return train_losses, val_losses, saved_model_file, run_name

