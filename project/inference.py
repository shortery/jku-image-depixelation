import torch
import torch.utils.data
import numpy as np

def predict(
    network: torch.nn.Module,
    test_set_dict: dict,
    batch_size: int,
    device: str
    ) -> np.ndarray:
    """Compute predictions"""

    pixelated_array = np.asarray(test_set_dict["pixelated_images"], dtype=np.float32)
    known_array = np.asarray(test_set_dict["known_arrays"], dtype=np.float32)

    # normalize
    pixelated_array /= 255
    known_array /= 255

    # create dataloader
    concat_pixelated_known = np.concatenate((pixelated_array, known_array), axis=1)
    dataloader = torch.utils.data.DataLoader(concat_pixelated_known, batch_size=batch_size, pin_memory=True)

    # compute predictions
    all_predictions = []
    with torch.no_grad():
        for concat_pixelated_known in dataloader:
            concat_pixelated_known = concat_pixelated_known.to(device)
            prediction = network(concat_pixelated_known)  # Get model prediction
            known_array = concat_pixelated_known[:, 1:2, :, :]
            for i in range(len(prediction)):
                output_sliced = prediction[i][known_array[i] == False]
                output_sliced = torch.clip(output_sliced * 255, min=0, max=255)
                output_sliced:torch.Tensor = output_sliced.to(torch.uint8).cpu().numpy()
                all_predictions.append(output_sliced)

    return all_predictions
