from .density import PointDensityNet


def get_network(name, num_view=4):
    if name == "density":
        return PointDensityNet(num_view)
    else:
        raise Exception("Unsupport Network")
