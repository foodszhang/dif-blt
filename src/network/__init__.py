from .density import PointWiseFusionDensityNet


def get_network(name):
    if name == "density":
        return PointWiseFusionDensityNet
    else:
        raise Exception("Unsupport Network")
