import torch
import os
import pandas as pd
from torchvision import transforms
from glob import glob
import xml.etree.ElementTree as ET
import cv2
from torchvision import transforms
import random
from PIL import Image
class DataLoader(torch.utils.data.DataLoader):

    def __init__(self, dataset, **kwargs):
        """Accepts a :class:`detecto.core.Dataset` object and creates
        an iterable over the data, which can then be fed into a
        :class:`detecto.core.Model` for training and validation.
        Extends PyTorch's `DataLoader
        <https://pytorch.org/docs/stable/data.html>`_ class with a custom
        ``collate_fn`` function.
        :param dataset: The dataset for iteration over.
        :type dataset: detecto.core.Dataset
        :param kwargs: (Optional) Additional arguments to customize the
            DataLoader, such as ``batch_size`` or ``shuffle``. See `docs
            <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_
            for more details.
        :type kwargs: Any
        **Example**::
           # >>> from detecto.core import Dataset, DataLoader
           # >>> dataset = Dataset('labels.csv', 'images/')
           # >>> loader = DataLoader(dataset, batch_size=2, shuffle=True)
           # >>> for images, targets in loader:
           # >>>     print(images[0].shape)
           # >>>     print(targets[0])
            torch.Size([3, 1080, 1720])
            {'boxes': tensor([[884, 387, 937, 784]]), 'labels': 'person'}
            torch.Size([3, 1080, 1720])
            {'boxes': tensor([[   1,  410, 1657, 1079]]), 'labels': 'car'}
            ...
        """

        super().__init__(dataset, collate_fn=DataLoader.collate_data, **kwargs)

    # Converts a list of tuples into a tuple of lists so that
    # it can properly be fed to the model for training
    @staticmethod
    def collate_data(batch):
        images, targets = zip(*batch)
        return list(images), list(targets)

class Dataset(torch.utils.data.Dataset):

    def __init__(self, label_data, image_folder=None, transform=None):
        """Takes in the path to the label data and images and creates
        an indexable dataset over all of the data. Applies optional
        transforms over the data. Extends PyTorch's `Dataset
        <https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset>`_.
        :param label_data: Can either contain the path to a folder storing
            the XML label files or a CSV file containing the label data.
            If a CSV file, the file should have the following columns in
            order: ``filename``, ``width``, ``height``, ``class``, ``xmin``,
            ``ymin``, ``xmax``, and ``ymax``. See
            :func:`detecto.utils.xml_to_csv` to generate CSV files in this
            format from XML label files.
        :type label_data: str
        :param image_folder: (Optional) The path to the folder containing the
            images. If not specified, it is assumed that the images and XML
            files are in the same directory as given by `label_data`. Defaults
            to None.
        :type image_folder: str
        :param transform: (Optional) A torchvision `transforms.Compose
            <https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.Compose>`__
            object containing transformations to apply on all elements in
            the dataset. See `PyTorch docs
            <https://pytorch.org/docs/stable/torchvision/transforms.html>`_
            for a list of possible transforms. When using transforms.Resize
            and transforms.RandomHorizontalFlip, all box coordinates are
            automatically adjusted to match the modified image. If None,
            defaults to the transforms returned by
            :func:`detecto.utils.default_transforms`.
        :type transform: torchvision.transforms.Compose or None
        **Indexing**:
        A Dataset object can be indexed like any other Python iterable.
        Doing so returns a tuple of length 2. The first element is the
        image and the second element is a dict containing a 'boxes' and
        'labels' key. ``dict['boxes']`` is a torch.Tensor of size
        ``(1, 4)`` containing ``xmin``, ``ymin``, ``xmax``, and ``ymax``
        of the box and ``dict['labels']`` is the string label of the
        detected object.
        **Example**::
       #     >>> from detecto.core import Dataset
       #     >>> # Create dataset from separate XML and image folders
       #     >>> dataset = Dataset('xml_labels/', 'images/')
       #     >>> # Create dataset from a combined XML and image folder
       #     >>> dataset1 = Dataset('images_and_labels/')
       #     >>> # Create dataset from a CSV file and image folder
       #     >>> dataset2 = Dataset('labels.csv', 'images/')
       #     >>> print(len(dataset))
       #     >>> image, target = dataset[0]
       #     >>> print(image.shape)
       #     >>> print(target)
            4
            torch.Size([3, 720, 1280])
            {'boxes': tensor([[564, 43, 736, 349]]), 'labels': 'balloon'}
        """

        # CSV file contains: filename, width, height, class, xmin, ymin, xmax, ymax
        if os.path.isfile(label_data):
            self._csv = pd.read_csv(label_data)
        else:
            self._csv = xml_to_csv(label_data)

        # If image folder not given, set it to labels folder
        if image_folder is None:
            self._root_dir = label_data
        else:
            self._root_dir = image_folder

        if transform is None:
            self.transform = default_transforms()
        else:
            self.transform = transform

    # Returns the length of this dataset
    def __len__(self):
        return len(self._csv)

    # Is what allows you to index the dataset, e.g. dataset[0]
    # dataset[index] returns a tuple containing the image and the targets dict
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Read in the image from the file name in the 0th column
        img_name = os.path.join(self._root_dir, self._csv.iloc[idx, 0])
        image = read_image(img_name)

        # Read in xmin, ymin, xmax, and ymax
        box = self._csv.iloc[idx, 4:]
        box = torch.tensor(box).view(1, 4)

        # Read in the label
        label = self._csv.iloc[idx, 3]

        targets = {'boxes': box, 'labels': label}

        # Perform transformations
        if self.transform:
            width = self._csv.loc[idx, 'width']
            height = self._csv.loc[idx, 'height']

            # Apply the transforms manually to be able to deal with
            # transforms like Resize or RandomHorizontalFlip
            updated_transforms = []
            scale_factor = 1.0
            random_flip = 0.0
            for t in self.transform.transforms:
                # Add each transformation to our list
                updated_transforms.append(t)

                # If a resize transformation exists, scale down the coordinates
                # of the box by the same amount as the resize
                if isinstance(t, transforms.Resize):
                    original_size = min(height, width)
                    scale_factor = original_size / t.size

                # If a horizontal flip transformation exists, get its probability
                # so we can apply it manually to both the image and the boxes.
                elif isinstance(t, transforms.RandomHorizontalFlip):
                    random_flip = t.p

            # Apply each transformation manually
            for t in updated_transforms:
                # Handle the horizontal flip case, where we need to apply
                # the transformation to both the image and the box labels
                if isinstance(t, transforms.RandomHorizontalFlip):
                    if random.random() < random_flip:
                        image = transforms.RandomHorizontalFlip(1)(image)
                        # Flip box's x-coordinates
                        box[0, 0] = width - box[0, 0]
                        box[0, 2] = width - box[0, 2]
                        box[0, 0], box[0, 2] = box[0, (2, 0)]
                else:
                    # if not (isinstance(image, Image.Image)):
                    #     image = Image.fromarray(image)
                    image = t(image)

            # Scale down box if necessary
            targets['boxes'] = (box / scale_factor[0]).long() # TODO

        return image, targets

def xml_to_csv(xml_folder, output_file=None):
    """Converts a folder of XML label files into a pandas DataFrame and/or
    CSV file, which can then be used to create a :class:`detecto.core.Dataset`
    object. Each XML file should correspond to an image and contain the image
    name, image size, and the names and bounding boxes of the objects in the
    image, if any. Extraneous data in the XML files will simply be ignored.
    See :download:`here <../_static/example.xml>` for an example XML file.
    For an image labeling tool that produces XML files in this format,
    see `LabelImg <https://github.com/tzutalin/labelImg>`_.
    :param xml_folder: The path to the folder containing the XML files.
    :type xml_folder: str
    :param output_file: (Optional) If given, saves a CSV file containing
        the XML data in the file output_file. If None, does not save to
        any file. Defaults to None.
    :type output_file: str or None
    :return: A pandas DataFrame containing the XML data.
    :rtype: pandas.DataFrame
    **Example**::
       # >>> from detecto.utils import xml_to_csv
       # >>> # Saves data to a file called labels.csv
       # >>> xml_to_csv('xml_labels/', 'labels.csv')
       # >>> # Returns a pandas DataFrame of the data
       # >>> df = xml_to_csv('xml_labels/')
    """

    xml_list = []
    # Loop through every XML file
    for xml_file in glob(xml_folder + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        filename = root.find('filename').text
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        # Each object represents each actual image label
        for member in root.findall('object'):
            box = member.find('bndbox')
            label = member.find('name').text

            # Add image file name, image size, label, and box coordinates to CSV file
            row = (filename, width, height, label, int(float(box[0].text)),
                   int(float(box[1].text)), int(float(box[2].text)), int(float(box[3].text)))
            xml_list.append(row)

    # Save as a CSV file
    column_names = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_names)

    if output_file is not None:
        xml_df.to_csv(output_file, index=None)

    return xml_df

def read_image(path):
    """Helper function that reads in an image as a
    `NumPy <https://numpy.org/>`_ array. Equivalent to using
    `OpenCV <https://docs.opencv.org/master/>`_'s cv2.imread
    function and converting from BGR to RGB format.
    :param path: The path to the image.
    :type path: str
    :return: Image in NumPy array format
    :rtype: ndarray
    **Example**::
      #  >>> import matplotlib.pyplot as plt
      #  >>> from detecto.utils import read_image
      #  >>> image = read_image('image.jpg')
      #  >>> plt.imshow(image)
      #  >>> plt.show()
    """

    image = cv2.imread(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def default_transforms():
    """Returns the default, bare-minimum transformations that should be
    applied to images passed to classes in the :mod:`detecto.core` module.
    :return: A torchvision `transforms.Compose
        <https://pytorch.org/docs/stable/torchvision/transforms.html>`_
        object containing a transforms.ToTensor object and the
        transforms.Normalize object returned by
        :func:`detecto.utils.normalize_transform`.
    :rtype: torchvision.transforms.Compose
    **Example**::
     #   >>> from detecto.core import Dataset
     #   >>> from detecto.utils import default_transforms
     #   >>> # Note: if transform=None, the Dataset will automatically
     #   >>> # apply these default transforms to images
     #   >>> defaults = default_transforms()
     #   >>> dataset = Dataset('labels.csv', 'images/', transform=defaults)
    """

    return transforms.Compose([transforms.ToTensor(), normalize_transform()])

def normalize_transform():
    """Returns a torchvision `transforms.Normalize
    <https://pytorch.org/docs/stable/torchvision/transforms.html>`_ object
    with default mean and standard deviation values as required by PyTorch's
    pre-trained models.
    :return: A transforms.Normalize object with pre-computed values.
    :rtype: torchvision.transforms.Normalize
    **Example**::
      #  >>> from detecto.core import Dataset
      #  >>> from detecto.utils import normalize_transform
      #  >>> from torchvision import transforms
      #  >>> # Note: if transform=None, the Dataset will automatically
      #  >>> # apply these default transforms to images
      #  >>> defaults = transforms.Compose([
      #  >>>     transforms.ToTensor(),
      #  >>>     normalize_transform(),
      #  >>> ])
      #  >>> dataset = Dataset('labels.csv', 'images/', transform=defaults)
    """

    # Default for PyTorch's pre-trained models
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def filter_top_predictions(labels, boxes, scores):
    """Filters out the top scoring predictions of each class from the
    given data. Note: passing the predictions from
    :meth:`detecto.core.Model.predict` to this function produces the same
    results as a direct call to :meth:`detecto.core.Model.predict_top`.
    :param labels: A list containing the string labels.
    :type labels: list
    :param boxes: A tensor of size [N, 4] containing the N box coordinates.
    :type boxes: torch.Tensor
    :param scores: A tensor containing the score for each prediction.
    :type scores: torch.Tensor
    :return: Returns a tuple of the given labels, boxes, and scores, except
        with only the top scoring prediction of each unique label kept in;
        all other predictions are filtered out.
    :rtype: tuple
    **Example**::
      #  >>> from detecto.core import Model
      #  >>> from detecto.utils import read_image, filter_top_predictions
      #  >>> model = Model.load('model_weights.pth', ['label1', 'label2'])
      #  >>> image = read_image('image.jpg')
      #  >>> labels, boxes, scores = model.predict(image)
      #  >>> top_preds = filter_top_predictions(labels, boxes, scores)
      #  >>> top_preds
        (['label2', 'label1'], tensor([[   0.0000,  428.0744, 1617.1860, 1076.3607],
        [ 875.3470,  412.1762,  949.5915,  793.3424]]), tensor([0.9397, 0.8686]))
    """

    filtered_labels = []
    filtered_boxes = []
    filtered_scores = []
    # Loop through each unique label
    for label in set(labels):
        # Get first index of label, which is also its highest scoring occurrence
        index = labels.index(label)

        filtered_labels.append(label)
        filtered_boxes.append(boxes[index])
        filtered_scores.append(scores[index])

    if len(filtered_labels) == 0:
        return filtered_labels, torch.empty(0, 4), torch.tensor(filtered_scores)
    return filtered_labels, torch.stack(filtered_boxes), torch.tensor(filtered_scores)