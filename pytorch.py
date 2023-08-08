import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class pytorchdataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = pd.read_csv("training_data.csv", index_col = 0)

    def __getitem__(self, index):
        image_path = f"cleaned_images/{self.data.loc[index, 'id']}.jpg"
        label = self.data.loc[index, "labels"]
        index = self.data.loc[index, "id"]
        image = Image.open(image_path)
        resize = transforms.Resize((256, 256))
        image = resize(image)
        convert_tensor = transforms.ToTensor()
        features = convert_tensor(image)
        return ((features, label, index))
        
    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    dataset = pytorchdataset()
    features, label, index = dataset[0]
