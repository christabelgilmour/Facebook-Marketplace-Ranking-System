from torchvision import transforms


def transform_image(image):
    resize = transforms.Resize((256, 256))
    convert_tensor = transforms.ToTensor()
    image = resize(image)
    image = convert_tensor(image)
    transformed_image = image.unsqueeze(0)
    return transformed_image
