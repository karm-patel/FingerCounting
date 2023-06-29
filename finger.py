import tensorflow as tf
import tensorflow_datasets as tfds
import glob

label_dict = {}
for i in range(12):
  if i <=5:
    label_dict[i] = str(i)+'L'
    continue
  label_dict[i] = str(i-6)+'R'

label_dict_flipped = {value: key for key, value in label_dict.items()}

# label_dict

DATASET_DIR = "dataset_test"
NUM_CLASSES = 12
class Fingers(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for fingersataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Dataset metadata (homepage, citation,...)."""
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=(128, 128, 3)),
            'label': tfds.features.ClassLabel(
                names=[str(i) for i in range(NUM_CLASSES)],
                doc='Whether 0-5 corresponds to left hand and 6-11 corresponds to right hand'),
        }),
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Download the data and define splits."""
#     extracted_path = dl_manager.download_and_extract('http://data.org/data.zip')
    # dl_manager returns pathlib-like objects with `path.read_text()`,
    # `path.iterdir()`,...
    
    return {
        'train': self._generate_examples(path=f'{DATASET_DIR}/train'),
        'test': self._generate_examples(path=f'{DATASET_DIR}/test'),
    }

  def _generate_examples(self, path):
    """Generator of examples for each split."""
    # import pdb; pdb.set_trace();    
    for img_path in glob.glob(f'{path}/*.png'):
      label = label_dict_flipped[img_path.split("_")[-1].split(".")[0]]
      # Yields (key, example)
      yield img_path, {
          'image': img_path,
          'label': label,
      }