from .density import PointDensityNet


def get_network(name):
    if name == "density":
        return PointDensityNet
    else:
        raise Exception("Unsupport Network")
