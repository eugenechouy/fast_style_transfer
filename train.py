import argparse
import numpy as np
import torch
import os
import utils

from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from model import TransformerNet
from model import Vgg16

def train(args):
    device = torch.device("cpu")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load dataset
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    # load style image
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    style = utils.load_image(args.style_image, size=args.style_size)
    style = style_transform(style)
    style = style.repeat(args.batch_size, 1, 1, 1).to(device)

    # load network
    transformer = TransformerNet().to(device)
    vgg = Vgg16(requires_grad=False).to(device)

    # define optimizer and loss function
    optimizer = Adam(transformer.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()

    features_style = vgg(utils.normalize_batch(style))
    gram_style = [utils.gram_matrix(y) for y in features_style]

    for e in range(args.epochs):
        transformer.train()
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            count += len(x)
            optimizer.zero_grad()

            image_original = x.to(device)
            image_transformed = transformer(x)

            image_original = utils.normalize_batch(image_original)
            image_transformed = utils.normalize_batch(image_transformed)

            # extract features for compute content loss
            features_original= vgg(image_original)
            features_transformed = vgg(image_transformed)
            content_loss = args.content_weight * mse_loss(features_transformed.relu3_3, features_original.relu3_3)

             # extract features for compute style loss
            style_loss = 0.
            for ft_y, gm_s in zip(features_transformed, gram_style):
                gm_y = utils.gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:len(x), :, :])
            style_loss *= args.style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            if (batch_id + 1) % 200 == 0:
                print("Epoch {}:[{}/{}]".format(e + 1, count, len(train_dataset)))

    # save model
    transformer.eval().cpu()
    save_model_filename = "epoch_" + str(args.epochs) + "_" + str(args.content_weight) + "_" + str(args.style_weight) + ".model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(transformer.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    arg_parser.add_argument("--epochs", type=int, default=2)
    arg_parser.add_argument("--batch-size", type=int, default=4)
    arg_parser.add_argument("--dataset", type=str, required=True)
    arg_parser.add_argument("--style-image", type=str, default="images/style_image/udnie.jpg")
    arg_parser.add_argument("--save-model-dir", type=str, default="models/")
    arg_parser.add_argument("--image-size", type=int, default=256)
    arg_parser.add_argument("--style-size", type=int, default=None)
    arg_parser.add_argument("--seed", type=int, default=42)
    arg_parser.add_argument("--content-weight", type=float, default=1e5)
    arg_parser.add_argument("--style-weight", type=float, default=1e10)
    arg_parser.add_argument("--lr", type=float, default=1e-3)

    args = arg_parser.parse_args()
    
    train(args)