from sustainbench import get_dataset
from sustainbench.common.data_loaders import get_train_loader
import torchvision.transforms as transforms

dataset = get_dataset(dataset='dhs_dataset')
train_data = dataset.get_subset('train')
train_loader = get_train_loader('standard', train_data, batch_size=16)