
from hyperbox.networks.resnet import resnet20

if __name__ == '__main__':
    size = lambda net: sum([p.numel() for name, p in net.named_parameters() if 'value' not in name])
    net = resnet20()
    params_num = size(net)
    mb = params_num * 4 / 1024**2
    print(f"#pamrams of {net.__class__.__name__}={size(net)} {mb}(MB)")
