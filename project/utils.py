import numpy as np

def to_grayscale(pil_image: np.ndarray) -> np.ndarray:
    """Convert pil_image to grayscale using the colorimetric conversion"""
    image_shape = pil_image.shape
    if len(image_shape) == 2:
        return pil_image.reshape(1, *(image_shape)).copy()
    elif len(image_shape) == 3:
        if image_shape[-1] != 3:
            raise ValueError("The third dimension must be 3.")
    else:
        raise ValueError("The image has a wrong shape.")
    
    normalized_image = pil_image / 255
    RGB_linear = []
    for i in range(3):
        C = normalized_image[:,:,i]
        RGB_linear.append(np.where(C <= 0.04045, C / 12.92, ((C + 0.055) / 1.055) ** 2.4))
    Y_linear = 0.2126 * RGB_linear[0] + 0.7152 * RGB_linear[1] + 0.0722 * RGB_linear[2]
    Y = np.where(Y_linear <= 0.0031308, 12.92 * Y_linear, 1.055 * (Y_linear ** (1 / 2.4)) - 0.055)
    Y_out = Y * 255
    if not np.issubdtype(Y_out.dtype, np.integer):
        Y_out = Y_out.round(decimals=0)
    Y_out = Y_out.astype(pil_image.dtype)
    Y_out = Y_out.reshape(1, *(Y_out.shape))

    return Y_out




