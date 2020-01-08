import argparse
import os
import sys
import re
import torch
import utils

from torchvision import transforms
from model import TransformerNet

def stylize(args):
    device = torch.device("cpu")

    content_image = utils.load_image(args.content_image, scale=args.content_scale)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    with torch.no_grad():
        style_model = TransformerNet()
        state_dict = torch.load(args.model)
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        output = style_model(content_image)
       
    utils.save_image(args.output_image, output[0])

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")

    arg_parser.add_argument("--content-image", type=str, required=True)
    arg_parser.add_argument("--content-scale", type=float, default=None)
    arg_parser.add_argument("--output-image", type=str, required=True)
    arg_parser.add_argument("--model", type=str, required=True)

    args = arg_parser.parse_args()

    stylize(args)
