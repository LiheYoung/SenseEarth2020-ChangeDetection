from models.sseg.deeplabv3plus import DeepLabV3Plus
from models.sseg.fcn import FCN
from models.sseg.pspnet import PSPNet


def get_model(model, backbone, pretrained, nclass, lightweight):
    if model == "fcn":
        model = FCN(backbone, pretrained, nclass, lightweight)
    elif model == "pspnet":
        model = PSPNet(backbone, pretrained, nclass, lightweight)
    elif model == "deeplabv3plus":
        model = DeepLabV3Plus(backbone, pretrained, nclass, lightweight)
    else:
        exit("\nError: MODEL \'%s\' is not implemented!\n" % model)

    params_num = sum(p.numel() for p in model.parameters())
    print("\nParams: %.1fM" % (params_num / 1e6))

    return model
