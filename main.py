import numpy as np
import torch

import commentjson as json

import tinycudann as tcnn

from tqdm import tqdm


class NeRF(torch.nn.Module):
    def __init__(self, encoding, network_rgb, network_sigma):
        super(NeRF, self).__init__()

        self.encoding = encoding
        self.network_rgb = network_rgb
        self.network_sigma = network_sigma

    def forward(self, x):
        
        x_enc = self.encoding(x)
        x_rgb = self.network_rgb(x_enc)
        x_sigma = self.network_sigma(x_enc)

        return x_rgb, x_sigma



if __name__ == '__main__':

    with open('./configs/config.json', 'r') as f:
        config = json.load(f)
    
    # Combined network (should be faster if a second network can be added)
    # model = tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=4, encoding_config=config["encoding"], network_config=config["network"])

    encoding = tcnn.Encoding(n_input_dims=3, encoding_config=config['encoding'])
    network_rgb = tcnn.Network(n_input_dims=encoding.n_output_dims, n_output_dims=3, network_config=config['network_rgb'])
    network_sigma = tcnn.Network(n_input_dims=encoding.n_output_dims, n_output_dims=1, network_config=config['network_sigma'])

    model = NeRF(encoding, network_rgb, network_sigma)

    x = np.random.rand(1024*512, 3)
    x = torch.Tensor(x).cuda()

    for i in tqdm(range(10000)):
        with torch.no_grad():
            y_rgb, y_sigma = model(x)

    print(y_rgb.shape, y_sigma.shape)