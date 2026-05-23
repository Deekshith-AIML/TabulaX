import torch
print(torch.cuda.is_available())  # Should return True if CUDA is detected
print(torch.cuda.get_device_name(0))  # Prints the GPU name
print(torch.version.cuda)  # Prints the CUDA version used by PyTorch
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))  # Lists GPU devices