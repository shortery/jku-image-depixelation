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



def prepare_image(
    image: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create the pixelated image by pixelating a rectangular area
    Args:
        image: 3D numpy array with shape (1, H, W) that contains a grayscale image
        x, y: x- and y-coordinates within image where the pixelated area should start
        widts, height: width and height of the pixelated area
        size: block size of the pixelation process
    Returns:
        3-tuple (pixelated_image, known_array, target_array)
        pixelated_image: 3D numpy array that represents the pixelated version of the input image. Shape (1, H, W)
        known_array: boolean 3D numpy array that has entries True for all original (unchanged) pixels and False for unknown (pixelated) pixels. Shape (1, H, W)
        target_array: 3D numpy array that represents the original pixels of the pixelated area before pixelation process
    """
    
    if len(image.shape) != 3:
        raise ValueError("The image must have three dimensions.")
    channel, H, W = image.shape
    if channel != 1:
        raise ValueError("The image must have the first dimension (the channel size) equal to 1.")
    if width < 2 or height < 2 or size < 2:
        raise ValueError("Width, height and size must be greater than or equal to 2.")
    if x < 0 or y < 0:
        raise ValueError("The x and y coordinates must be greater than or equal to 0.")
    if x + width > W:
        raise ValueError("The pixelated area exceeded the input image width.")
    if y + height > H:
        raise ValueError("The pixelated area exceeded the input image height.")

    pix_indexes = slice(None, None), slice(y, y+height), slice(x, x+width)

    pixelated_image = image.copy()
    pixelated_image[pix_indexes] = pixelate_window(pixelated_image[pix_indexes], size)

    known_array = np.ones(image.shape, dtype=bool)
    known_array[pix_indexes] = False

    target_array = image[pix_indexes].copy()

    return pixelated_image, known_array, target_array



def pixelate_window(window:np.ndarray, size: int) -> np.ndarray:
    """Pixelate a rectangular area (window)"""
    if len(window.shape) != 3 or window.shape[0] != 1:
        raise ValueError("Expected shape (1, H, W).")
    
    window = window.reshape(window.shape[-2:])
    pix_window = np.ones(window.shape).astype(window.dtype)
    max_r = np.ceil(window.shape[0]/size).astype(int)
    max_c = np.ceil(window.shape[1]/size).astype(int)
    for r in range(max_r):
        for c in range(max_c):
            indexes = slice(size*r, size*(r+1)), slice(size*c, size*(c+1))
            block = window[indexes]
            pix_window[indexes] = block.mean()
    return pix_window.reshape(1, *(pix_window.shape))
