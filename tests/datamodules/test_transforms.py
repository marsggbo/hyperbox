from hyperbox.datamodules.transforms import get_transforms, TorchTransforms

if __name__ == '__main__':
    kwargs = {
        'input_size': [32, 32],
        'random_crop': {'enable': 1, 'padding': 4, 'size': 28},
        'random_horizontal_flip': {'enable': 1, 'p': 0.8}
    }
    T = TorchTransforms(**kwargs)
    pass