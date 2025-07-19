import os
from pathlib import Path
from time import time, sleep, ctime
from rich import print
from PIL import Image
import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torchvision.transforms import ToTensor, Resize, Lambda


"""
NOTE add to readme in repo

cassette beasts source: https://wiki.cassettebeasts.com/wiki/Species
digimon source: https://www.kaggle.com/datasets/hjjznb/digimon-images-dataset
dragon quest source: https://game8.co/games/DQM-Dark-Prince/archives/435873
palworld source: https://palworld.fandom.com/wiki/Palpedia#Regular
SMT images source: https://megatenwiki.com/wiki/Gallery:Persona_5#Artwork_2
pokemon source: https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types?select=images

GOAL WITH fan_art FOLDER
I wanna collect images of fan-created designs and see if this model can guess the game their inspired from "correctly."
It's basically using AI to judge if a fan design is similar to the original's art style I guess.

"""


# CONSTANTS
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# File
DATA_DIRECTORY = 'data'
MODELS_DIRECTORY = 'models'
FANART_DIRECTORY = 'fan_art'

# Data
CHANNELS = 3
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32

# Defines preprocessing data transformations
# Images passed to the model have to undergo these transformations or it won't work  
DATA_TRANSFORMS = transforms.Compose([
    Lambda(lambda img: img.convert('RGB')),
    Resize(IMAGE_SIZE),
    ToTensor(),
    nn.Flatten(),
])


# CLASSES
class ImageClassifier(nn.Module):
    """
    Defines the neural network structure.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            
            # Model expects 256x256 images in RGB format == 256 * 256 * 3(r, g, and b) number of parameters
            nn.Linear(IMAGE_SIZE[0]*IMAGE_SIZE[1] * CHANNELS, 512),
            
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, len(get_label_names())),
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# FUNCTIONS
def get_label_names() -> list[str]:
    return [name for name in os.listdir(DATA_DIRECTORY) if os.path.isdir(os.path.join(DATA_DIRECTORY, name))]


def get_dataset(fol: os.PathLike | str = DATA_DIRECTORY) -> ImageFolder:
    return ImageFolder(
        fol,
        transform=DATA_TRANSFORMS,
        loader=Image.open,
        allow_empty=True
    )


def get_model() -> ImageClassifier:
    return ImageClassifier().to(DEVICE)


def training_loop(
        m: nn.Module, 
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn = nn.CrossEntropyLoss() 
        ) -> None:

    # Setting model to training mode
    m.train()

    for batch, (X, y) in enumerate(dataloader):

        # Converting tensors to current device
        X, y = X.to(DEVICE), y.to(DEVICE)
        
        # Compute prediction and loss
        pred = m(X)
        loss = loss_fn(pred, y)

        # Backpropogation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{len(dataloader.dataset):>5d}]")

        # print(f"Loss: {loss.item()}")


def testing_loop(m: nn.Module, dataloader: DataLoader, loss_fn = nn.CrossEntropyLoss()) -> None:
    
    # Set model to evaluation mode
    m.eval()

    loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:

            # Convert tensors to current device
            X, y = X.to(DEVICE), y.to(DEVICE)

            # Testing model
            pred = m(X)
            loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    loss /= len(dataloader)
    correct /= len(dataloader.dataset)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {loss:>8f} \n")


def train_model(
        m: nn.Module, 
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        num_epochs: int = 8,
        learning_rate: float = 1e-3,
        loss_fn = nn.CrossEntropyLoss() ) -> None:

    # Get time before training
    start_time = time()
    
    # Set optimizer
    optim = torch.optim.SGD(m.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        print(f'Epoch #{epoch+1}\n-------------------------------')
        training_loop(m, train_dataloader, optimizer=optim, loss_fn=loss_fn)
        testing_loop(m, eval_dataloader, loss_fn)

    # Display the training time
    elapsed_time = time() - start_time
    print(f'Training finished in {clock_time(elapsed_time)}')


def save_model(m: nn.Module, name: os.PathLike | Path | str) -> None:
    os.makedirs(MODELS_DIRECTORY, exist_ok=True)
    
    # Validating name
    working_path = os.path.join(MODELS_DIRECTORY, name)
    if '.' in working_path: 
        assert ('.pt' in working_path) or ('.pth' in working_path)
    else:
        working_path += '.pt'
    
    # Validating parent folder
    parent_dir = Path(working_path).parent
    assert os.path.exists(parent_dir), f'Parent directory not found: {parent_dir}'

    assert not (os.path.isfile(working_path) or os.path.isdir(working_path))

    with open(working_path, 'wb') as f:
        torch.save(m, f)    


def load_model(path: nn.Module | os.PathLike | Path | str) -> nn.Module:
    if isinstance(path, nn.Module):
        return path
    
    # Validating path
    assert os.path.exists(path), f'Invalid path: {path}'
    assert os.path.isfile(path), f'Invalid item; not a file: {path}'
    assert path.endswith('.pt') or path.endswith('.pth'), f'Invalid file type: must be ".pt" or ".pth": {path}'
    
    # m = ImageClassifier().to(DEVICE)
    # m.load_state_dict(torch.load(path, weights_only=True))
    # return m

    return torch.load(path, weights_only=False)


def split_data(
        dataset: Dataset, 
        weights: tuple[float] = [0.8, 0.1, 0.1], 
        batch_size: int = BATCH_SIZE 
        ) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Splits dataset into 3 Dataloaders for training, evaluation, and testing dataloaders respectively

    Args:
        dataset - source dataset
        weights - list of weights to determine data allocation
        batch_size - batch size for loaders
    
    Returns:
        Tuple of training, evaluation, and testing dataloaders respectively
    
    """
    
    # Split data
    train_data, eval_data, test_data = random_split(
        dataset, lengths=weights
    )
    
    # Creating loaders
    train_dl = DataLoader(train_data, batch_size=batch_size)
    eval_dl = DataLoader(eval_data, batch_size=batch_size)
    test_dl = DataLoader(test_data, batch_size=batch_size)

    return train_dl, eval_dl, test_dl


def preprocess_image(img: Image.Image| os.PathLike | str) -> torch.Tensor:
    img_data = img if isinstance(img, Image.Image) else Image.open(img)
    assert img_data != None, f'Failed to read image: {img}'
    return DATA_TRANSFORMS(img_data).unsqueeze(0).to(DEVICE)


def clear_console():
    """Clears the console screen based on the operating system."""
    # For Windows
    if os.name == 'nt':
        _ = os.system('cls')
    # For macOS and Linux
    else:
        _ = os.system('clear')


def clock_time(seconds: float) -> str:
    """
    Convert seconds to 00:00 clock time.
    source: https://stackoverflow.com/questions/27496889/converting-a-float-to-hhmm-format
    """
    return '{0:02.0f}:{1:02.0f}'.format(*divmod(seconds * 60, 60))


def train_new_model(
        name: str, 
        num_epochs: int = 8,
        learning_rate: float = 1e-3,
        loss_fn = nn.CrossEntropyLoss() ) -> None:
    
    clear_console()
    print(f'[yellow]Using {DEVICE} device')

    print('\nGetting data ...')
    sleep(1)
    dataset = get_dataset(DATA_DIRECTORY)

    print(dataset)

    print('\nSplitting data ...')
    sleep(1)

    # In tensorflow, 'testing' data would be saved until the end after all the
    # other training and evaluation loops, but idk if I'm gonna implement that
    training, evaluation, testing = split_data(dataset, batch_size=BATCH_SIZE)
    
    print(f'\tTraining: {len(training)}')
    print(f'\tEvaluation: {len(evaluation)}')
    print(f'\tTesting: {len(testing)}')
    
    print('\nBuilding model')
    sleep(1.5)
    
    model = get_model()
    print(model)
    
    print('\nTraining model')
    sleep(0.5)

    train_model(model, training, evaluation, num_epochs, learning_rate, loss_fn)
    
    # Final test using data the model hasn't seen yet
    print('\nTesting model ...')
    testing_loop(model, testing)
    
    save_model(model, name)


def make_prediction(
        m: nn.Module | os.PathLike | Path | str, 
        img: Image.Image | os.PathLike | Path | str
        ) -> str:
    """
    Use a model to make a prediction. Returns the predicted franchise of the input image

    Args:
        m - path to model or a nn.Module object
        img - path to image file or Image.Image object
    
    Returns:
        String of the predicted class' name
    """
    
    # Getting model
    loaded_model: nn.Module
    if isinstance(m, nn.Module):
        loaded_model = m
    else:
        loaded_model = load_model(m)
    assert(loaded_model != None)
    
    # Parsing image data
    img_data = preprocess_image(img)
    assert(img_data != None)

    loaded_model.eval()
    classes = get_label_names()
    with torch.no_grad():
        pred = loaded_model(img_data)
        label_name = classes[pred[0].argmax(0)]
    
    return label_name


def test_fan_art(m: nn.Module) -> None:
    """
    Tests the fan art folder. Displays predicted and actual results for each image.
    """

    raise NotImplementedError()
    
    start_time = time()
    print('\n[yellow]TESTING FAN ART')
    
    print('\nLoading model')
    loaded_model = load_model(m)
    print(loaded_model)

    print('Loading data')
    # dataloader = DataLoader(data, batch_size= BATCH_SIZE)

    # Doesn't use testing_loop because the print statements are different
    print('\nRunning tests ...')

    # Setting model to evaluation state
    loaded_model.eval()
    
    classes = get_label_names()
    with torch.no_grad():

        """
        I actually don't wanna use a dataloader because I want to load individual images and print out their paths/names.
        I'm gonna need to go through each subfolder in DATA_DIRECTORY, read individual image files, preprocesses them, then feed
        them to the model.
        
        """
        
        pass


if __name__ == '__main__':
    # TODO cli or summ

    # Testing single, unsorted image
    # test_imgpath = r'unsorted_images\Mimikyu.png'
    # model = load_model(r'models\mon_categorizer1_2.pt')
    # prediction = make_prediction(model, test_imgpath)
    # print(f'Predicted that {os.path.basename(test_imgpath)} is a {prediction.capitalize()}')

    # Testing loading dataset
    # data = get_dataset(DATA_DIRECTORY)
    # print(data)
    
    # Creating new model
    train_new_model('mon_categorizer1_2.pt', num_epochs=20)
    
    pass
    