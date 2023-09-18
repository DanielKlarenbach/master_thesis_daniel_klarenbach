# data preprocessing

import os
import nibabel as nib
import itertools
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import grid, remove_self_loops
from torchvision.transforms import Resize, Normalize
import torch_geometric
from PIL import Image
import numpy as np
from torchvision.transforms.functional import  InterpolationMode
import sys
import time
import math

"""
Parameters
"""
IMAGE_SIZE = (50, 50)

class InRAMMemoryDataset(InMemoryDataset):
  def __init__(
    self,
    root,
    labels_dir,
    get_image_object,
    get_image,
    resize_images,
  ):
    self.labels_dir = labels_dir
    self.get_image_object = get_image_object
    self.get_image = get_image
    self.resize_images = resize_images

    super().__init__(root)

    self.data, self.slices = torch.load(self.processed_paths[0])

  @property
  def raw_file_names(self):
    return os.listDir(self.raw_dir) # will never download

  @property
  def processed_file_names(self):
    return 'data.pt'

  def process(self):
    file_paths = list_dir_files(self.raw_dir)

    images = self.get_images(file_paths)
    print(f'Number of images: {len(images)}')
    sys.stdout.flush()

    resized_images = [
      resize_image(image) for image in images
    ]
    print(f'Number of resized images: {len(resized_images)}')
    sys.stdout.flush()

    normalized_images = normalize_images(resized_images)
    print(f'Number of normalized images: {len(images)}')
    sys.stdout.flush()

    zipped_pyg_input = zip(file_paths, resized_images, normalized_images)

    pyg_data_list = []
    for input_index, pyg_input in enumerate(zipped_pyg_input):
      if input_index % 50 == 0:
        print(f'Processed {input_index}/{len(file_paths)} images')
        sys.stdout.flush()

      file_path, resized_image, normalized_image = pyg_input
      pyg_data_list.append(
        to_pyg_Data(self.labels_dir, file_path, resized_image, normalized_image)
      )
    
    pyg_data_list = [
      pyg_data for pyg_data in pyg_data_list if pyg_data is not None
    ]
    print(f'Number of pyg data objects: {len(pyg_data_list)}')
    sys.stdout.flush()

    data, slices = self.collate(pyg_data_list)
    torch.save((data, slices), self.processed_paths[0])

  def get_images(self, file_paths):
    image_objects = [
      self.get_image_object(file_path) for file_path in file_paths
    ]

    if self.get_image is not None:
      images = [
        self.get_image(image_object) for image_object in image_objects
      ]
      # flatten
      images = list(itertools.chain(*images))
    else:
      images = image_objects

    return images

def nii_get_image_object(file_path):
  nii_image_proxy = nib.load(file_path)
  return nii_image_proxy.get_fdata()

def image_get_image_object(file_path):
  image = Image.open(file_path)
  return np.asarray(image)

def nii_get_image(image_object):
  two_dimensional_images = []
  for two_dimensional_image_index in range(image_object.shape[2]):
    two_dimensional_images.append(
      image_object[:, : ,two_dimensional_image_index]
    )

  return two_dimensional_images

def to_pyg_Data(labels_dir, file_path, image, normalized_image):
  image_size = (len(image), len(image[0]))
  (row, col), pos = grid(height=image_size[0], width=image_size[1])

  edge_index_with_self_loops = \
    torch.tensor([row.numpy(), col.numpy()], dtype=torch.long)
  edge_index_without_self_loops = \
    remove_self_loops(edge_index_with_self_loops)[0]

  node_feature_vectors = [
    pixel_intensities.tolist()
      for row in normalized_image
        for pixel_intensities in row
  ]
  x = torch.tensor(node_feature_vectors)

  try:
    labels_file_name = os.path.basename(os.path.splitext(file_path)[0])
    labels_file_path = labels_dir + '/' + labels_file_name + '.png'

    labels_image = Image.open(labels_file_path)
    labels_image = np.asarray(labels_image)

    segments_labels = resize_labels(labels_image)
  except Exception as e:
    return None

  pyg_data = Data(
    x = x,
    segments_labels = segments_labels,
    image = image,
    edge_index = edge_index_without_self_loops
  )
  pyg_data.validate(raise_on_error = True)

  return pyg_data

def normalize_images(images):
  org_images = torch.stack(images)

  compute_matrics_images = torch.permute(org_images, (3, 0, 1, 2))
  compute_matrics_images = torch.flatten(compute_matrics_images, start_dim = 1)

  means = torch.mean(compute_matrics_images, dim = -1)
  stds = torch.std(compute_matrics_images, dim = -1)
  print(f'Dataset means before normalization: {means}')
  print(f'Datatset stds before normalization: {stds}')
  sys.stdout.flush()

  norm_transform = Normalize(means, stds)
  transformed_images = torch.permute(org_images, (0, 3, 1, 2))
  transformed_images = torch.stack(
    [norm_transform(image) for image in transformed_images]
  )
  transformed_images = torch.permute(transformed_images, (0, 2, 3, 1))

  print(
    f'''Dataset means after normalization: 
      {torch.mean(
        torch.flatten(
          torch.permute(transformed_images, (3, 0, 1, 2)),
          start_dim = 1
        ),
        -1
      )}'''
  )
  print(
    f'''Dataset stds after normalization: 
      {torch.std(
        torch.flatten(
          torch.permute(transformed_images, (3, 0, 1, 2)),
          start_dim = 1
        ),
        -1
      )}'''
  )
  sys.stdout.flush()

  return transformed_images

def resize_image(unresized_image):
  unresized_image = torch.tensor(unresized_image, dtype=float)
  unresized_image = torch.permute(unresized_image, (2, 0, 1))

  resize_transform = Resize(
    IMAGE_SIZE, interpolation = InterpolationMode.BILINEAR
  )
  resized_image = resize_transform(unresized_image)
  resized_image = torch.permute(resized_image, (1, 2, 0))

  return resized_image

def resize_labels(unresized_labels):
  unresized_labels = torch.tensor(unresized_labels, dtype=float)
  unresized_labels = torch.unsqueeze(unresized_labels, 0)

  resize_transform = Resize(
    IMAGE_SIZE, interpolation = InterpolationMode.NEAREST
  )
  resized_labels = resize_transform(unresized_labels)
  resized_labels = torch.squeeze(resized_labels, 0)

  return resized_labels

def list_dir_files(directory):
  return [
    os.path.abspath(os.path.join(dirpath, file_name))
      for dirpath, _, filenames in os.walk(directory)
        for file_name in filenames
  ]

# sacroiliac joints dataset

"""
Parameters
"""
BASE_DIR_PATH = '/net/tscratch/people/plgdklarenbach/VOC'

print('Started preprocessing dataset')
sys.stdout.flush()

dataset = InRAMMemoryDataset(
  root = BASE_DIR_PATH,
  labels_dir = BASE_DIR_PATH + '/labels',
  get_image_object = image_get_image_object,
  get_image = None,
  resize_images = True
)

# network architecture

import torch
from torch.nn.functional import relu
from torch.nn import BatchNorm1d, ModuleList
from torch_geometric.nn import SAGEConv, dense_diff_pool, dense_mincut_pool, DMoNPooling, MLP
from torch_geometric.nn import aggr
from torch_geometric.utils import to_dense_adj, to_dense_batch

"""
Parameters
"""
SUPERPIXELS_COUNT = 100
HIDDEN_LAYERS_COUNT = 3
NODE_EMBEDDING_SIZE = 200

layers_aggregations = [
  aggr.MeanAggregation(),
  aggr.MinAggregation(),
  aggr.MaxAggregation(),
  aggr.StdAggregation(),
  aggr.VarAggregation()
]

class GNN(torch.nn.Module):
  def __init__(self):
    super(GNN, self).__init__()

    self.conv_layers = ModuleList()
    self.norm_layers = ModuleList()

    for layer_index in range(HIDDEN_LAYERS_COUNT):
      if layer_index == 0:
          in_channels = 3
      else:
          in_channels = NODE_EMBEDDING_SIZE
      
      self.conv_layers.append(
        SAGEConv(
          in_channels = -1,
          out_channels = NODE_EMBEDDING_SIZE,
          aggr = aggr.MultiAggregation(
            aggrs = layers_aggregations,
            mode = 'proj',
            mode_kwargs = dict(in_channels = in_channels, out_channels = NODE_EMBEDDING_SIZE)
          ),
          normalize = True,
        )
      )

      self.norm_layers.append(
        BatchNorm1d(num_features = NODE_EMBEDDING_SIZE)
      )

    self.out_conv = SAGEConv(
      in_channels = -1,
      out_channels = NODE_EMBEDDING_SIZE,
      aggr = aggr.MultiAggregation(
        aggrs = layers_aggregations,
        mode = 'proj',
        mode_kwargs = dict(in_channels = NODE_EMBEDDING_SIZE, out_channels = NODE_EMBEDDING_SIZE)
      ),
      normalize = True,
    )
    self.out_norm = BatchNorm1d(num_features = NODE_EMBEDDING_SIZE)

  def forward(self, x, edge_index):
    for layer_index in range(HIDDEN_LAYERS_COUNT):
      x = self.conv_layers[layer_index](x, edge_index)
      x = x.relu()
      x = self.norm_layers[layer_index](x)

    x = self.out_conv(x, edge_index)
    x = x.relu()
    x = self.out_norm(x)

    return x

class Model(torch.nn.Module):
  def __init__(self):
    super(Model, self).__init__()

    self.gnn = GNN()
    self.classifier = MLP([
        NODE_EMBEDDING_SIZE,
        SUPERPIXELS_COUNT
    ])
    self.pool1 = DMoNPooling([NODE_EMBEDDING_SIZE, SUPERPIXELS_COUNT], SUPERPIXELS_COUNT)

  def forward(self, x, edge_index, batch):
    x = self.gnn(x, edge_index)
    x, mask = to_dense_batch(x, batch)
    adj = to_dense_adj(edge_index, batch)

    class_probs, _, _, l1, _, l2 = self.pool1(x, adj, mask)

    return class_probs, l1 * (1000) + l2, l2

from collections import defaultdict
from skimage.measure import label

@torch.no_grad()
def compute_undersegmentation_error_and_superpixels_sizes_std(superpixels_labels, segments_labels):
  superpixels_labels = torch.flatten(superpixels_labels)
  segments_labels = torch.flatten(segments_labels)

  unique_superpixels_labels, pixels_per_superpixel = \
    torch.unique(superpixels_labels, return_counts = True)
  superpixels_sizes = {
    unique_superpixels_labels[superpixel_index].item():
    pixels_per_superpixel[superpixel_index].item()
      for superpixel_index in range(len(unique_superpixels_labels))
  }

  unique_segments_labels = \
    torch.unique(segments_labels)
  segments_p_in = {
    unique_segments_labels[segment_index].item():
    defaultdict(int)
      for segment_index in range(len(unique_segments_labels))
  }
  segments_p_out = {
    unique_segments_labels[segment_index].item():
    defaultdict(int)
      for segment_index in range(len(unique_segments_labels))
  }

  # compute p_in
  for pixel_index in range(IMAGE_SIZE[0] * IMAGE_SIZE[1]):
      segments_p_in[segments_labels[pixel_index].item()][superpixels_labels[pixel_index].item()] +=1

  # compute p_out
  for segment_key in segments_p_in:
    for superpixel_key in segments_p_in[segment_key]:
      segments_p_out[segment_key][superpixel_key] = \
        superpixels_sizes[superpixel_key] - segments_p_in[segment_key][superpixel_key]

  underseg_error = 0
  for segment_key in segments_p_in:
    # omit borders
    if segment_key == 255:
      continue
    for superpixel_key in segments_p_in[segment_key]: 
      p_in = segments_p_in[segment_key][superpixel_key]
      p_out =  segments_p_out[segment_key][superpixel_key]
      underseg_error += min(p_in, p_out)
  underseg_error = underseg_error / (IMAGE_SIZE[0] * IMAGE_SIZE[1])

  return underseg_error, np.var([*superpixels_sizes.values()]) * len(unique_superpixels_labels)

@torch.no_grad()
def compute_intra_cluster_variation(image, superpixels_labels):
  image = torch.flatten(image, end_dim = -2)
  superpixels_labels = torch.flatten(superpixels_labels)
  
  unique_superpixels_labels = torch.unique(superpixels_labels)
  superpixels_pixels = {
    unique_superpixels_labels[i].item(): 
    {0: [], 1: [], 2:[]} 
      for i in range(len(unique_superpixels_labels))
  }

  for pixel_index in range(IMAGE_SIZE[0] * IMAGE_SIZE[1]):
    for channel_ind in range(3):
      superpixels_pixels[superpixels_labels[pixel_index].item()][channel_ind].append(
        image[pixel_index][channel_ind]
      )

  superpixels_variations = []
  for channel_index in range(3):
    channels_variations = []

    for superpixel_key in superpixels_pixels:
      pixels = torch.tensor(
        superpixels_pixels[superpixel_key][channel_ind], dtype = float
      )
      pixels_mean = torch.mean(
        pixels
      )
      nominator = torch.sqrt(
        torch.sum((pixels - pixels_mean) ** 2)
      )

      channels_variations.append(
        nominator / len(pixels)
      )

    channels_variations = torch.tensor(channels_variations, dtype=float)
    superpixels_variations.append(
      torch.mean(channels_variations)
    )

  superpixels_variations = torch.tensor(superpixels_variations)

  return torch.sum(superpixels_variations)

@torch.no_grad()
def compute_disjoint_metrics(superpixels_labels):
  relabeled_superpixels_labels = torch.flatten(
    torch.tensor(
      label(superpixels_labels.cpu().numpy(), connectivity=2)
    )
  )
  unique_relabeled_superpixels = torch.unique(relabeled_superpixels_labels)
  superpixels_labels = torch.flatten(superpixels_labels)
  unique_superpixels_labels = \
    torch.unique(superpixels_labels)

  disjoint_counts = {
    unique_superpixels_labels[superpixel_index].item():
    []
      for superpixel_index in range(len(unique_superpixels_labels))
  }

  for pixel_index in range(IMAGE_SIZE[0] * IMAGE_SIZE[1]):
    disjoint_counts[superpixels_labels[pixel_index].item()].append(
        relabeled_superpixels_labels[pixel_index]
    )

  disjointed_superpixels_count = 0
  disjointed_superpixels_parts = []
  for superpixel_key in disjoint_counts:
    unique_counts = len(np.unique(disjoint_counts[superpixel_key]))
    disjoint_counts[superpixel_key] = unique_counts

    if unique_counts != 1:
      disjointed_superpixels_parts.append(unique_counts)
      disjointed_superpixels_count += 1

  return disjointed_superpixels_count, len(unique_relabeled_superpixels),  len(unique_superpixels_labels)

@torch.no_grad()
def batch_compute_matrics(
  images_batch,
  superpixels_labels_batch,
  segments_labels_batch
):
  batch_under_error = 0
  batch_superpixels_sizes_std = 0
  batch_inter_variance = 0
  batch_disjointed_percentage = 0
  batch_disjointed_mean_parts = 0
  superpixels = 0

  for batch_index in range(len(images_batch)):
    image = images_batch[batch_index]
    superpixels_labels = superpixels_labels_batch[batch_index]
    segments_labels = segments_labels_batch[batch_index]

    under_error, superpixels_sizes_std = compute_undersegmentation_error_and_superpixels_sizes_std(
      superpixels_labels, segments_labels
    )
    batch_under_error += under_error
    batch_superpixels_sizes_std += superpixels_sizes_std

    batch_inter_variance += compute_intra_cluster_variation(image, superpixels_labels)
    
    disjointed_percentage, disjointed_mean_parts, d_superpixels = compute_disjoint_metrics(
      superpixels_labels
    )
    batch_disjointed_percentage += disjointed_percentage
    batch_disjointed_mean_parts += disjointed_mean_parts
    superpixels += d_superpixels

  return (
    batch_under_error,
    batch_superpixels_sizes_std,
    batch_inter_variance,
    batch_disjointed_percentage,
    batch_disjointed_mean_parts,
    superpixels
  )

# training

from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import numpy as np
import torch
from matplotlib import pyplot as plt

"""
Parameters
"""
BATCH_SIZE = 32
LEARNING_RATE = 0.001
OPTIMIZER = torch.optim.Adam
EPOCHS_COUNT = 500

torch.manual_seed(0)
train_dataset, val_dataset, test_dataset = random_split(
        dataset, [0.7, 0.2, 0.1]
)
print(f'Number of training samples: {len(train_dataset)}')
print(f'Number of validation samples: {len(val_dataset)}')
print(f'Number of testing samples: {len(test_dataset)}')
sys.stdout.flush()

train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")
sys.stdout.flush()

model = Model().to(device)
optimizer = OPTIMIZER(model.parameters(), lr=LEARNING_RATE)

def train():
  model.train()
  epoch_loss = 0
  l1_e = 0
  l2_e = 0

  underseg_error = 0
  superpixels_sizes_std = 0
  inter_variation = 0
  disjointed_superpixels_percentage = 0
  disjointed_parts_mean = 0

  for batch_index, data in enumerate(train_dataloader):
    torch.cuda.empty_cache()
    data = data.to(device)
    preds, l1, l2 = model(data.x, data.edge_index, data.batch)
    loss = l1

    batches_count = len(train_dataloader)
    print(f"Batch {batch_index+1}/{batches_count} loss: {loss}")
    sys.stdout.flush()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    current_batch_size = int(data.x.size(0) / (IMAGE_SIZE[0] * IMAGE_SIZE[1]))
    epoch_loss += current_batch_size * loss.item()
    l1_e += l1.item() * current_batch_size
    l2_e += l2.item() * current_batch_size
    '''
    image_batch = torch.reshape(
      data.image, (current_batch_size, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    )

    superpixels_labels_batch = torch.argmax(preds, -1)
    superpixels_labels_batch = torch.reshape(
      superpixels_labels_batch,
      (current_batch_size, IMAGE_SIZE[0], IMAGE_SIZE[1])
    )

    segments_labels_batch = torch.reshape(
      data.segments_labels, (current_batch_size, IMAGE_SIZE[0], IMAGE_SIZE[1])
    )
    batch_under_error, batch_superpixels_sizes_std, batch_inter_variance, batch_disjointed_percentage, batch_disjointed_mean_parts = \
      batch_compute_matrics(
        image_batch, superpixels_labels_batch, segments_labels_batch
      )

    underseg_error += batch_under_error
    superpixels_sizes_std += batch_superpixels_sizes_std
    inter_variation += batch_inter_variance
    disjointed_superpixels_percentage += batch_disjointed_percentage
    disjointed_parts_mean += batch_disjointed_mean_parts
    '''

  return (epoch_loss / len(train_dataloader.dataset), 
    underseg_error / len(train_dataloader.dataset),
    superpixels_sizes_std / len(train_dataloader.dataset),
    inter_variation / len(train_dataloader.dataset),
    disjointed_superpixels_percentage / len(train_dataloader.dataset),
    disjointed_parts_mean / len(train_dataloader.dataset),
    l1_e   / len(train_dataloader.dataset),
    l2_e / len(train_dataloader.dataset)
  )

@torch.no_grad()
def train_test(dataloader):
  model.eval()
  epoch_loss = 0

  underseg_error = 0
  superpixels_sizes_std = 0
  inter_variation = 0
  disjointed_superpixels_percentage = 0
  disjointed_parts_mean = 0
  superpixels = 0

  for data in dataloader:
    data = data.to(device)
    preds, l1, l2 = model(data.x, data.edge_index, data.batch)
    loss = l1

    current_batch_size = int(data.x.size(0) / (IMAGE_SIZE[0] * IMAGE_SIZE[1]))
    epoch_loss += current_batch_size * loss.item()

  return (epoch_loss / len(dataloader.dataset),
    underseg_error / len(train_dataloader.dataset),
    superpixels_sizes_std / len(dataloader.dataset),
    inter_variation / len(dataloader.dataset),
    disjointed_superpixels_percentage / len(dataloader.dataset),
    disjointed_parts_mean / len(dataloader.dataset)
  )

@torch.no_grad()
def test(dataloader):
  model.eval()
  epoch_loss = 0

  underseg_error = 0
  superpixels_sizes_std = 0
  inter_variation = 0
  disjointed_superpixels_percentage = 0
  disjointed_parts_mean = 0
  superpixels = 2
  l1_e = 0
  l2_e = 0

  for data in dataloader:
    data = data.to(device)
    preds, l1, l2 = model(data.x, data.edge_index, data.batch)
    loss = l1
    current_batch_size = int(data.x.size(0) / (IMAGE_SIZE[0] * IMAGE_SIZE[1]))

    l1_e += current_batch_size * (l1 - l2).item()
    l2_e += current_batch_size * l2.item()
    print(loss)
    sys.stdout.flush()
    
    epoch_loss += current_batch_size * loss.item()
    
  return (epoch_loss / len(dataloader.dataset),
    underseg_error / len(dataloader.dataset),
    math.sqrt(superpixels_sizes_std / superpixels),
    inter_variation / superpixels,
    disjointed_superpixels_percentage / superpixels,
    disjointed_parts_mean / superpixels,
    superpixels,
    l1_e / len(dataloader.dataset),
    l2_e / len(dataloader.dataset),
  )


def plot(
  first_plot_data,
  second_plot_data,
  chart_title,
  y_label,
  first_plot_title,
  second_second_title
):

  plt.plot(first_plot_data, label = first_plot_title)
  plt.plot(second_plot_data, label = second_second_title)

  plt.title(chart_title)
  plt.xlabel('Epoch')
  plt.ylabel(y_label)

  plt.legend(loc='upper right')
    
  plt.savefig(f'{sys.argv[1]}/{chart_title}.png')
  plt.show()
  plt.clf()
  sys.stdout.flush()

def plot_with_different_scales(
  first_plot_data,
  second_plot_data,
  chart_title,
  first_y_label,
  second_y_label,
  first_plot_title,
  second_second_title
):

  fig, ax1 = plt.subplots()
  ax1.set_title(chart_title)

  color = '#1f77b4'

  ax1.set_xlabel('Epoch')
  ax1.set_ylabel(first_y_label, color = color)
  ax1.plot(first_plot_data, color = color, label = first_plot_title)
  ax1.tick_params(axis='y', labelcolor = color)

  ax2 = ax1.twinx()
  color = '#ff7f0e'
  ax2.set_ylabel(second_y_label, color = color)
  ax2.plot(second_plot_data, color = color, label = second_second_title)
  ax2.tick_params(axis='y', labelcolor = color)

  plt.savefig(f'{sys.argv[1]}/{chart_title}.png')
  plt.show()
  plt.clf()
  sys.stdout.flush()

train_epoch_losses = []
validation_epoch_losses = []

train_underseg_errors = []
val_underseg_errors = []

train_superpixels_sizes_std = []
val_superpixels_sizes_std = []

train_inter_variation = []
val_inter_variation = []

train_disjointed_superpixels_percentage = []
val_disjointed_superpixels_percentage = []

train_disjointed_parts_mean = []
val_disjointed_parts_mean = []

best_val_epoch_loss = 10000000

l1_a =[]
l2_a = []
'''
for epoch in range(EPOCHS_COUNT):
  print(f"Epoch {epoch+1}\n-------------------------------")
  (train_epoch_loss,
   train_under_error,
   train_sizes_std,
   train_inter_var,
   train_disjointed_per,
   train_disjointed_parts, l1, l2) = train()

  (validation_epoch_loss,
   val_under_error,
   val_sizes_std,
   val_inter_var,
   val_disjointed_per,
   val_disjointed_parts) = train_test(val_dataloader)
  sys.stdout.flush()

  
  print(
    f"Epoch {epoch+1}  Train loss: {train_epoch_loss}  Val loss: {validation_epoch_loss}"
  )
  sys.stdout.flush()

  if validation_epoch_loss < best_val_epoch_loss:
    print(f'Saving best model, epoch: {epoch+1}')
    sys.stdout.flush()
    torch.save(model.state_dict(), f'{sys.argv[1]}/trained_model.pt')
    best_val_epoch_loss = validation_epoch_loss

  train_epoch_losses.append(train_epoch_loss)
  validation_epoch_losses.append(validation_epoch_loss)

  train_underseg_errors.append(train_under_error)
  val_underseg_errors.append(val_under_error)

  train_superpixels_sizes_std.append(train_sizes_std)
  val_superpixels_sizes_std.append(val_sizes_std)
  
  train_inter_variation.append(train_inter_var)
  val_inter_variation.append(val_inter_var)

  train_disjointed_superpixels_percentage.append(train_disjointed_per)
  val_disjointed_superpixels_percentage.append(val_disjointed_per)

  train_disjointed_parts_mean.append(train_disjointed_parts)
  val_disjointed_parts_mean.append(val_disjointed_parts)
  l1_a.append(l1)
  l2_a.append(l2)

plot(
  train_epoch_losses,
  validation_epoch_losses,
  'Training and validation loss',
  'Loss',
  'Training',
  'Validation'
)

plot(
  train_underseg_errors,
  val_underseg_errors,
  'Training and validation undersegmentation error',
  'Undersegmentation error',
  'Training',
  'Validation'
)

plot(
  train_superpixels_sizes_std,
  val_superpixels_sizes_std,
  'Training and validation superpixel size standard deviation',
  'Superpixel size standard deviation',
  'Training',
  'Validation'
)

plot(
  train_inter_variation,
  val_inter_variation,
  'Training and validation intra-cluster variation',
  'Intra-cluster variation',
  'Training',
  'Validation'
)

plot(
  np.array(train_disjointed_superpixels_percentage) * 100,
  np.array(val_disjointed_superpixels_percentage) * 100,
  'Training and validation percentage of disconnected superpixels',
  'Percentage of disconnected superpixels [%]',
  'Training',
  'Validation'
)

plot(
  train_disjointed_parts_mean,
  val_disjointed_parts_mean,
  'Training and validation mean number of superpixel parts',
  'Mean number of superpixel parts',
  'Training',
  'Validation'
)

plot_with_different_scales(
  l1_a,
  l2_a,
  'Cut and orthogonality loss',
  'Cut loss',
  'Orthogonality loss',
  'Cut loss',
  'Orthogonality loss',
)

plot_with_different_scales(
  train_epoch_losses,
  train_superpixels_sizes_std,
  'Training loss and superpixel size standard deviation',
  'Loss',
  'Superpixel size standard deviation',
  'Loss',
  'Superpixel size standard deviation'
)

plot_with_different_scales(
  train_epoch_losses,
  train_inter_variation,
  'Training loss and intra-cluster variation',
  'Loss',
  'Intra-cluster variation',
  'Loss',
  'Intra-cluster variation'
)

plot_with_different_scales(
  train_epoch_losses,
  np.array(train_disjointed_superpixels_percentage) * 100,
  'Training loss and percentage of disconnected superpixels',
  'Loss',
  'Percentage of disconnected superpixels',
  'Loss',
  'Percentage of disconnected superpixels'
)

plot_with_different_scales(
  train_epoch_losses,
  train_disjointed_parts_mean,
  'Training loss and mean number of superpixel parts',
  'Loss',
  'Mean number of superpixel parts',
  'Loss',
  'Mean number of superpixel parts'
)
'''
from skimage.segmentation import slic
model = Model().to(device)
model.load_state_dict(torch.load(f'{sys.argv[1]}/trained_model.pt'))
model.eval()
print('Model loaded')
sys.stdout.flush()

(test_epoch_loss,
  test_under_error,
  test_sizes_std,
  test_inter_var,
  test_disjointed_per,
  test_disjointed_parts,
  test_superpixels,
  t_l1,
  t_l2
) = test(test_dataloader)

print('Test metrics:')
print(
  test_epoch_loss,
  test_under_error,
  test_sizes_std,
  test_inter_var,
  test_disjointed_per,
  test_disjointed_parts,
  test_superpixels,
  t_l1,
  t_l2
)
sys.stdout.flush()

# results visual overview

import numpy as np
from matplotlib import pyplot as plt
from skimage import color
import networkx as nx

def display_results(dataset, start_index, end_index):
  for index, data in enumerate(dataset[start_index:end_index]):
    data = data.to(device)
    data.x = data.x.float()
    predictions, _, _ = model(data.x, data.edge_index, data.batch)

    plt.figure(figsize=(20, 5))

    plt.subplot(151)
    image = data.image.cpu().numpy().astype(int)
    plt.imshow(image)
    plt.title("Image")

    plt.subplot(152)
    predicted_superpixel_lables = \
      predictions_to_labels(predictions).astype(dtype=np.int32)
    colored_predicted_superpixel_lables = color.label2rgb(
        predicted_superpixel_lables,
        image,
        kind='avg',
        bg_label=None
    )
    plt.imshow(colored_predicted_superpixel_lables)
    plt.title("Generated superpixels")

    plt.subplot(153)
    relabeled_superpixels_labels = label(predicted_superpixel_lables, connectivity=2)
    colored_relabeled_predicted_superpixel_lables = color.label2rgb(
        relabeled_superpixels_labels,
        image,
        kind='avg',
        bg_label=None
    )
    plt.imshow(colored_relabeled_predicted_superpixel_lables)
    plt.title("Generated relabeled superpixels")

    plt.subplot(154)
    slic_superpixels_labels = \
        slic(
          image,
          n_segments = SUPERPIXELS_COUNT,
          compactness = 10,
        )
    colored_slic_superpixel_lables = color.label2rgb(
        slic_superpixels_labels,
        image,
        kind='avg',
        bg_label=None
    )
    plt.imshow(colored_slic_superpixel_lables)
    plt.title("SLIC's superpixels")

    plt.subplot(155)
    segments_labels = data.segments_labels.cpu().numpy().astype(int)
    plt.imshow(color.label2rgb(segments_labels))
    plt.title("Segments")
    plt.savefig(f'{sys.argv[1]}/visual_result_{index}.png')

print('Done!')
