from models import ResNetEncoder, ImageEncoder
import numpy as np
from utils import get_model_size

encoder = ResNetEncoder(num_filters=128, num_res_block=10)
print(encoder)
encoder_size = get_model_size(encoder)

encoder = ImageEncoder()
print(encoder)
encoder_size = get_model_size(encoder)