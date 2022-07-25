from os.path import join

_BASE_DATA_PATH = "../data"

dataset_config = {
    'mnist': {
        'path': join(_BASE_DATA_PATH, 'mnist'),
        'normalize': ((0.1307,), (0.3081,)),
        # Use the next 3 lines to use MNIST with a 3x32x32 input
        # 'extend_channel': 3,
        # 'pad': 2,
        # 'normalize': ((0.1,), (0.2752,))    # values including padding
    },
    'svhn': {
        'path': join(_BASE_DATA_PATH, 'svhn'),
        'resize': (28, 28),
        'crop': None,
        'flip': False,
        'normalize': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    },
}
for dset in dataset_config.keys():
    for k in ['resize', 'pad', 'crop', 'normalize', 'class_order', 'extend_channel']:
        if k not in dataset_config[dset].keys():
            dataset_config[dset][k] = None
    if 'flip' not in dataset_config[dset].keys():
        dataset_config[dset]['flip'] = False
