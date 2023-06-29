# -*- coding: utf-8 -*-

# data preprocessing

import os
import nibabel as nib
from skimage.segmentation import slic
import itertools
import torch
from torch_geometric.data import InMemoryDataset, Dataset, Data
from torch_geometric.utils import grid, remove_self_loops
from torchvision.transforms import Resize
import torch_geometric
from PIL import Image, ImageOps
import numpy as np
import sys
import time

start = time.time()

"""
Parameters
"""
SUPERPIXELS_COUNT = 100
DATASET_DIR = 'hearts'

# image size
images_for_size = os.listdir(f'/net/tscratch/people/plgdklarenbach/data/{DATASET_DIR}/{SUPERPIXELS_COUNT}/raw')
nii_image_for_size = nib.load(f'/net/tscratch/people/plgdklarenbach/data/{DATASET_DIR}/{SUPERPIXELS_COUNT}/raw/{images_for_size[0]}')
print(f'Getting the size of nii images from: {images_for_size[0]}')
IMAGE_SIZE = (nii_image_for_size.header.get_data_shape()[0], nii_image_for_size.header.get_data_shape()[1])
print(f'Size of images: from the {images_for_size[0]}: {IMAGE_SIZE}')

class InRAMMemoryDataset(InMemoryDataset):
  def __init__(
    self,
    root,
    get_image_object,
    grayscale_images=True,
    resize_images=False,
    get_image=None,
    transform=None,
    pre_transform=None,
    pre_filter=None
  ):
    self.get_image_object = get_image_object
    self.get_image = get_image
    self.grayscale_images = grayscale_images
    self.resize_images = resize_images

    super().__init__(root, transform, pre_transform, pre_filter)

    self.data, self.slices = torch.load(self.processed_paths[0])

  @property
  def raw_file_names(self):
    return os.listdir(self.raw_dir) # will never download

  @property
  def processed_file_names(self):
    return 'data.pt'

  def process(self):
    ungrayscaled_images = self.get_ungrayscaled_images()
    print(f'Number of images: {len(ungrayscaled_images)}')
    print(f'Image size: {len(ungrayscaled_images[0])} x {len(ungrayscaled_images[0][0])}')
    sys.stdout.flush()

    grayscaled_images = torch.tensor(grayscale_images(ungrayscaled_images)) \
      if self.grayscale_images else ungrayscaled_images
    print(f'Grayscaled {len(grayscaled_images)} images')
    sys.stdout.flush()

    resized_images = resize_images(grayscaled_images) \
      if self.resize_images else grayscaled_images
    print(f'Resized {len(resized_images)} images')
    sys.stdout.flush()
    
    processed_rezied_images_msg_threshold = 30
    data_list = []
    for (resized_image_index, resized_image) in enumerate(resized_images):
        if resized_image_index % processed_rezied_images_msg_threshold == 0:
            print(f'Processed {resized_image_index}/{len(resized_images)} of resized images')
            sys.stdout.flush()
        
        data_list.append(image_to_pyg_Data(resized_image))

    if self.pre_filter is not None:
      data_list = [data for data in data_list if self.pre_filter(data)]

    if self.pre_transform is not None:
      data_list = [self.pre_transform(data) for data in data_list]
    print(f'Transformed to pyg Data object {len(data_list)} images')
    sys.stdout.flush()

    data, slices = self.collate(data_list)
    torch.save((data, slices), self.processed_paths[0])

  def get_ungrayscaled_images(self):
    file_paths = absoluteListDir(self.raw_dir)
    image_objects = [
      self.get_image_object(file_path) for file_path in file_paths
    ]

    if self.get_image is not None:
      ungrayscaled_images = [
        self.get_image(image_object) for image_object in image_objects
      ]
      # flatten
      ungrayscaled_images = list(itertools.chain(*ungrayscaled_images))
    else:
      ungrayscaled_images = image_objects
    
    return ungrayscaled_images

class NotInMemoryDataset(Dataset):
  def __init__(
    self,
    root,
    get_image_object,
    grayscale_images=True,
    resize_images=True,
    get_image=None,
    transform=None,
    pre_transform=None,
    pre_filter=None
  ):
    self.get_image_object = get_image_object
    self.get_image = get_image
    self.grayscale_images = grayscale_images
    self.resize_images = resize_images

    super().__init__(root, transform, pre_transform, pre_filter)

  @property
  def raw_file_names(self):
    return os.listDir(self.raw_dir) # will never download

  @property
  def processed_file_names(self):
    ungrayscaled_images = self.get_ungrayscaled_images()
    data_file_names = [
      f'data_{image_index}.pt'
        for image_index in range(len(ungrayscaled_images))
    ]
    
    return data_file_names

  def process(self):
    ungrayscaled_images = self.get_ungrayscaled_images()
    
    grayscaled_images = torch.tensor(grayscale_images(ungrayscaled_images)) \
      if self.grayscale_images else ungrayscaled_images

    resized_images = resize_images(grayscaled_images) \
      if self.resize_images else grayscaled_images

    processed_rezied_images_msg_threshold = 10
    data_list = []
    for (resized_image_index, resized_image) in enumerate(resized_images):
        if resized_image_index / len(resized_images) * 100 >= processed_rezied_images_msg_threshold:
            processed_rezied_images_msg_threshold += 10
            print(f'Processed {processed_rezied_images_msg_threshold}% of resized images')
            sys.stdout.flush()
        data_list.append(image_to_pyg_Data(resized_image))

    if self.pre_filter is not None:
      data_list = [data for data in data_list if self.pre_filter(data)]

    if self.pre_transform is not None:
      data_list = [self.pre_transform(data) for data in data_list]

    for (data_index, data) in enumerate(data_list):
      torch.save(
        data, os.path.join(self.processed_dir, f'data_{data_index}.pt')
      )

  def get_ungrayscaled_images(self):
    file_paths = absoluteListDir(self.raw_dir)
    image_objects = [
      self.get_image_object(file_path) for file_path in file_paths
    ]

    if self.get_image is not None:
      ungrayscaled_images = [
        self.get_image(image_object) for image_object in image_objects
      ]
      # flatten
      ungrayscaled_images = list(itertools.chain(*ungrayscaled_images))
    else:
      ungrayscaled_images = image_objects

    return ungrayscaled_images

  def len(self):
    return len(self.processed_file_names)

  def get(self, idx):
    data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
    return data

def nii_get_image_object(file_path):
  nii_image_proxy = nib.load(file_path)
  print(f'Processing nii file: {file_path}, shape: {nii_image_proxy.header.get_data_shape()}')
  sys.stdout.flush()

  return nii_image_proxy.get_fdata()

def nii_get_image(image_object):
  two_dimensional_images = []

  for two_dimensional_image_index in range(image_object.shape[2]):
    two_dimensional_images.append(image_object[:,:,two_dimensional_image_index])

  return two_dimensional_images

def image_to_pyg_Data(image):
  image_size = (len(image), len(image[0]))
  (row, col), pos = grid(height=image_size[0], width=image_size[1])

  edge_index_with_self_loops = \
    torch.tensor([row.numpy(), col.numpy()], dtype=torch.long)
  edge_index_without_self_loops = \
    remove_self_loops(edge_index_with_self_loops)[0]

  node_feature_vectors = [
    [pixel_intensity, i+j] for i, row in enumerate(image) for j, pixel_intensity in enumerate(row)
  ]

  x = torch.tensor(node_feature_vectors, dtype=torch.float)
  superpixel_labels = \
    torch.from_numpy(
      slic(image, n_segments=SUPERPIXELS_COUNT, compactness=0.1, start_label=0)
    )
  y = torch.flatten(superpixel_labels)

  data = Data(x=x, y=y, edge_index=edge_index_without_self_loops)
  data.validate(raise_on_error=True)

  return data

def grayscale_images(ungrayscaled_images):
  grayscaled_images = [
    grayscale_image(unscaled_image)
      for unscaled_image in ungrayscaled_images
  ]

  return grayscaled_images

def bmp_get_image_object(file_path):
  return np.array(ImageOps.grayscale(Image.open(file_path)))

def grayscale_image(ungrayscaled_image):
  return (
    (ungrayscaled_image - ungrayscaled_image.min()) \
      * (1/(ungrayscaled_image.max() - ungrayscaled_image.min()) * 255)
  ).astype('uint8')

def resize_images(unresized_images):
  resize_transform = Resize(IMAGE_SIZE)
  resized_images = resize_transform(unresized_images)

  return resized_images

def absoluteListDir(directory):
  return [
    os.path.abspath(os.path.join(dirpath, f))
      for dirpath,_,filenames in os.walk(directory)
        for f in filenames
  ]

# sacroiliac joints dataset

print('Started preprocessing dataset')
sys.stdout.flush()

dataset = InRAMMemoryDataset(
  root=f'/net/tscratch/people/plgdklarenbach/data/{DATASET_DIR}/{SUPERPIXELS_COUNT}',
  get_image_object=nii_get_image_object,
  get_image=nii_get_image,
)
sys.stdout.flush()

# network architecture

import torch
from torch.nn.functional import relu
from torch.nn import BatchNorm1d, ModuleList
from torch_geometric.nn import ChebConv

"""
Parameters
"""
NODE_EMBEDDING_SIZE = 100

class Model(torch.nn.Module):
  def __init__(self):
    super(Model, self).__init__()

    self.convolutional_layers = ModuleList()
    self.batch_normalization_layers = ModuleList()

    self.convolutional_layers.append(
      ChebConv(
        in_channels=dataset.num_features,
        out_channels=NODE_EMBEDDING_SIZE,
        K=5
      )
    )
    self.batch_normalization_layers.append(
      BatchNorm1d(num_features=NODE_EMBEDDING_SIZE)
    )

    self.convolutional_layers.append(
      ChebConv(
        in_channels=NODE_EMBEDDING_SIZE,
        out_channels=NODE_EMBEDDING_SIZE,
        K=5
      )
    )
    self.batch_normalization_layers.append(
      BatchNorm1d(num_features=NODE_EMBEDDING_SIZE)
    )

    self.convolutional_layers.append(
      ChebConv(
        in_channels=NODE_EMBEDDING_SIZE,
        out_channels=NODE_EMBEDDING_SIZE,
        K=5
      )
    )
    self.batch_normalization_layers.append(
      BatchNorm1d(num_features=NODE_EMBEDDING_SIZE)
    )

    self.convolutional_layers.append(
      ChebConv(
        in_channels=NODE_EMBEDDING_SIZE,
        out_channels=NODE_EMBEDDING_SIZE,
        K=5
      )
    )
    self.batch_normalization_layers.append(
      BatchNorm1d(num_features=NODE_EMBEDDING_SIZE)
    )

    self.convolutional_layers.append(
       ChebConv(
        in_channels=NODE_EMBEDDING_SIZE,
        out_channels=SUPERPIXELS_COUNT,
        K=5
      )
    )
    self.batch_normalization_layers.append(
      BatchNorm1d(num_features=SUPERPIXELS_COUNT)
    )

  def forward(self, data):
    x, adj = data.x, data.edge_index
    for step in range(len(self.convolutional_layers)):
      convoluted_x = self.convolutional_layers[step](x, adj)
      relued_x = relu(convoluted_x)
      batch_norm_layer_output = \
        self.batch_normalization_layers[step](relued_x)
      x = batch_norm_layer_output

    return x

# monai loss function
from monai.losses.dice import *
import torch
import monai

def monai_loss_function(predicted_probabilities, target_labels):
  dice_loss = monai.losses.TverskyLoss(
      softmax=True,
  )
  loss = dice_loss(predicted_probabilities, target_labels)

  return loss

# training

from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt

"""
Parameters
"""
BATCH_SIZE = 8
LEARNING_RATE = 0.001
OPTIMIZER = torch.optim.Adam
EPOCHS_COUNT = 500
EARLY_STOPPING_EPOCHS = 30

train_dataset, val_dataset, test_dataset = random_split(
  dataset, [0.7, 0.2, 0.1]
)

train_dataloader = torch_geometric.loader.DataListLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
test_dataloader = torch_geometric.loader.DataListLoader(test_dataset, batch_size=BATCH_SIZE)
val_dataloader = torch_geometric.loader.DataListLoader(val_dataset, batch_size=BATCH_SIZE)

print(f"Number of avaiable cuda devices: {torch.cuda.device_count()}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")
print(torch.cuda.current_device())
sys.stdout.flush()

model = torch_geometric.nn.DataParallel(Model()).to(device)
optimizer = OPTIMIZER(model.parameters(), lr=LEARNING_RATE)

def cross_entropy_loss_function(predicted_probabilities, target_labels):
  loss = CrossEntropyLoss()(
    predicted_probabilities, target_labels
  )

  return loss

def train():
  model.train()
  epoch_loss = 0

  for batch_index, data in enumerate(train_dataloader):
    predicted_probabilities = model(data)
    y = torch.cat([d.y for d in data]).to(predicted_probabilities.device)

    loss = cross_entropy_loss_function(predicted_probabilities, y)
    batches_count = len(train_dataloader)
    print(f"Batch {batch_index+1}/{batches_count} loss: {loss}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    current_batch_size = y.size(0) / (IMAGE_SIZE[0] * IMAGE_SIZE[1])
    epoch_loss += current_batch_size * loss.item()

  return epoch_loss / len(train_dataset)

@torch.no_grad()
def test(dataloader):
  model.eval()
  epoch_loss = 0

  for data in dataloader:
    predicted_probabilities = model(data)
    y = torch.cat([d.y for d in data]).to(predicted_probabilities.device)

    loss = cross_entropy_loss_function(predicted_probabilities, y)

    current_batch_size = y.size(0) / (IMAGE_SIZE[0] * IMAGE_SIZE[1])
    epoch_loss += current_batch_size * loss.item()

  return epoch_loss / len(dataloader)

def plot_loss(actual_epochs, train_epoch_losses, validation_epoch_losses):
  epochs = np.arange(1, actual_epochs+1)

  plt.plot(epochs, train_epoch_losses, label='Training loss')
  plt.plot(epochs, validation_epoch_losses, label='Validation loss')

  plt.title('Training and validation loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')

  plt.xticks(np.arange(1, actual_epochs, 50))

  plt.legend(loc='best')
  plt.savefig(f'{sys.argv[1]}/training.png')

train_epoch_losses = []
validation_epoch_losses = []
best_val_epoch_loss = 1000000
early_stopping_count = EARLY_STOPPING_EPOCHS
actual_epochs = 0

for epoch in range(EPOCHS_COUNT):
  print(f"Epoch {epoch+1}\n-------------------------------")
  train_epoch_loss = train()
  validation_epoch_loss = test(val_dataloader)
  actual_epochs += 1

  print(
    f"Epoch {epoch+1}  Train loss: {train_epoch_loss:.4f}  Val loss: {validation_epoch_loss:.4f}"
  )

  train_epoch_losses.append(train_epoch_loss)
  validation_epoch_losses.append(validation_epoch_loss)

  if validation_epoch_loss < best_val_epoch_loss:
    print(f'Saving best model, epoch: {epoch+1}')
    torch.save(model.module.state_dict(), f'{sys.argv[1]}/trained_model.pt')
    best_val_epoch_loss = validation_epoch_loss
    early_stopping_count = EARLY_STOPPING_EPOCHS
  else:
    early_stopping_count -= 1

  if early_stopping_count == 0:
    break

  sys.stdout.flush()

test_loss = test(test_dataloader)
print(f"Test loss: {test_loss:.4f}")

plot_loss(actual_epochs, train_epoch_losses, validation_epoch_losses)

# results visual overview

import numpy as np
from torch_geometric.loader import DenseDataLoader
import matplotlib.pyplot as plt
from skimage import color

torch.set_printoptions(threshold=20)
np.set_printoptions(threshold=20)

loaded_model = Model()
loaded_model.load_state_dict(torch.load(f'{sys.argv[1]}/trained_model.pt'))
loaded_model.eval()
print('Model loaded')

def predictions_to_labels(batch_predictions):
  superpixel_labels = []

  for node_predictions in batch_predictions:
    max_pred_index = torch.argmax(node_predictions)
    superpixel_labels.append(max_pred_index.item())

  return np.reshape(superpixel_labels, IMAGE_SIZE)

def display_results(dataset, start_index, end_index):
  for (index, data) in enumerate(dataset[start_index:end_index]):
    data.x = data.x.float()
    predictions = loaded_model(data)

    x_numpy = data.x.cpu().numpy()[:,0]
    image = np.reshape(x_numpy, IMAGE_SIZE).astype(dtype=np.int32)

    plt.figure(figsize=(20, 5))

    plt.subplot(151)
    plt.imshow(image, cmap='inferno')
    plt.title("Original image")

    plt.subplot(152)
    target_superpixel_labels = torch.reshape(data.y, IMAGE_SIZE).cpu().numpy()
    plt.imshow(target_superpixel_labels, cmap='inferno')
    plt.title("Target superpixel labels")

    plt.subplot(153)
    predicted_superpixel_lables = \
      predictions_to_labels(predictions).astype(dtype=np.int32)
    print(target_superpixel_labels)
    print(predicted_superpixel_lables)
    plt.imshow(predicted_superpixel_lables, cmap='inferno')
    plt.title("Predicted superpixel labels")

    plt.subplot(154)
    colored_target_superpixel_labels = color.label2rgb(
        target_superpixel_labels,
        image,
        kind='avg',
        bg_label=0
    )
    plt.imshow(colored_target_superpixel_labels, cmap='inferno')
    plt.title("Colored target superpixel labels")

    plt.subplot(155)
    colored_predicted_superpixel_lables = color.label2rgb(
        predicted_superpixel_lables,
        image,
        kind='avg',
        bg_label=0
    )
    plt.imshow(colored_predicted_superpixel_lables, cmap='inferno')
    plt.title("Colored predicted superpixel labels")

    plt.savefig(f'{sys.argv[1]}/visual_result_{index}.png')


display_results(test_dataset, 0, 10)

end = time.time()
total_time = int((end - start) / 60)
print(f"Script runtime: {total_time} minutes")
