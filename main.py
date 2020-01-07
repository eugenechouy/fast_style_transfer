import argparse
import torch
import sys
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import datasets
from torchvision import transforms
from PIL import Image

from model import ImageTransformNet
from model import LossNetwork

def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img

def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std

def train(args):
    device = torch.device("cpu")
    torch.manual_seed(args.seed)

    # Create dataloader for the training data
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    # Load style image
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    style_image = load_image(args.style_image, size=args.style_size)
    style = style_transform(style_image)
    style = style.repeat(args.batch_size, 1, 1, 1).to(device)

    # Defines networks
    transform_net = ImageTransformNet().to(device)
    loss_net = LossNetwork().to(device)

    # Define optimizer and loss
    optimizer = Adam(transform_net.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss().to(device)

    # Extract style features
    features_style = loss_net(normalize_batch(style))
    gram_style = [gram_matrix(y) for y in features_style]

    for epoch in range(args.epochs):
        epoch_metrics = {"content": [], "style": [], "total": []}
        for batch_i, (images, _) in enumerate(train_loader):
            optimizer.zero_grad()

            images_original = images.to(device)
            images_transformed = transform_net(images_original)
		images_original = normalize_batch(images_original)
		images_transformed = normailize_batch(images_transformed)	    

            # Extract features
            features_original = loss_net(images_original)
            features_transformed = loss_net(images_transformed)

            # Compute content loss as MSE between features
            content_loss = args.content_weight * mse_loss(features_original.relu2_2, features_transformed.relu2_2)

            # Compute style loss as MSE between gram matrices
            style_loss = 0.
            for ft_y, gm_s in zip(features_transformed, gram_style):
                gm_y = gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:images.size(0), :, :])
            style_loss *= args.style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            epoch_metrics["content"] += [content_loss.item()]
            epoch_metrics["style"] += [style_loss.item()]
            epoch_metrics["total"] += [total_loss.item()]

            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [Content: %.2f (%.2f) Style: %.2f (%.2f) Total: %.2f (%.2f)]"
                % (
                    epoch + 1,
                    args.epochs,
                    batch_i,
                    len(train_dataset),
                    content_loss.item(),
                    np.mean(epoch_metrics["content"]),
                    style_loss.item(),
                    np.mean(epoch_metrics["style"]),
                    total_loss.item(),
                    np.mean(epoch_metrics["total"]),
                )
            )

def run(args):
    device = torch.device("cpu")

    content_image = load_image(args.content_image)
    content_transform = transforms.Compose([
        transforms.ToTensor(),                  # [0, 255] -> [0.0, 1.0]
        transforms.Lambda(lambda x: x.mul(255)) # mul 255
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        style_model = ImageTransformNet()
        state_dict = torch.load(args.model)
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        output = style_model(content_image).cpu()
    
    save_image(args.output_image, output[0])

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    subparsers = arg_parser.add_subparsers(title="subcommands", dest="subcommand")
    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--epochs", type=int, default=2)
    train_arg_parser.add_argument("--batch-size", type=int, default=4)
    train_arg_parser.add_argument("--dataset", type=str, required=True)
    train_arg_parser.add_argument("--style-image", type=str, default="images/style_image/udnie.jpg")
    train_arg_parser.add_argument("--save-model-dir", type=str, required=True)
    train_arg_parser.add_argument("--seed", type=int, default=42)
    train_arg_parser.add_argument("--lr", type=float, default=1e-3)
    train_arg_parser.add_argument("--content-weight", type=float, default=1e5)
    train_arg_parser.add_argument("--style-weight", type=float, default=1e10)
    train_arg_parser.add_argument("--image-size", type=int, default=256)
    train_arg_parser.add_argument("--style-size", type=int, default=None)

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument("--content-image", type=str, required=True)
    eval_arg_parser.add_argument("--output-image", type=str, required=True)
    eval_arg_parser.add_argument("--model", type=str, required=True)

    args = arg_parser.parse_args()
    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)
    if args.subcommand == "train":
        train(args)
    else:
        run(args)
