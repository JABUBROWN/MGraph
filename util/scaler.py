class NoScaler:

    def __init__(self, data, dim=1):
        self.dim = dim
        self.mean = data[..., :dim].mean()
        self.std = data[..., :dim].std()

    def scale(self, x):
        return x

    def reverse(self, x):
        return x


class NormalScaler:

    def __init__(self, data, dim=1):
        self.dim = dim
        self.mean = data[..., :dim].mean()
        self.std = data[..., :dim].std()

    def scale(self, x):
        x[..., :self.dim] = (x[..., :self.dim] - self.mean) / self.std
        return x

    def reverse(self, x):
        x[..., :self.dim] = (x[..., :self.dim] * self.std) + self.mean
        return x


# (X - min) / (max - min)
class ZeroOneScaler:

    def __init__(self, data, dim=1):
        self.dim = dim
        self.min_val = data[..., :self.dim].min()
        self.max_val = data[..., :self.dim].max()

    def scale(self, x):
        x[..., :self.dim] = (x[..., :self.dim] - self.min_val) / (self.max_val - self.min_val)
        return x

    def reverse(self, x):
        x[..., :self.dim] = x[..., :self.dim] * (self.max_val - self.min_val) + self.min_val
        return x


# [-1, 1]
class MinMaxScaler:

    def __init__(self, data, dim=1):
        self.dim = dim
        self.min_val = data[..., :self.dim].min()
        self.max_val = data[..., :self.dim].max()

    def scale(self, x):
        x[..., :self.dim] = (x[..., :self.dim] - self.min_val) / (self.max_val - self.min_val) * 2 - 1
        return x

    def reverse(self, x):
        x[..., :self.dim] = (x[..., :self.dim] + 1) / 2 * (self.max_val - self.min_val) + self.min_val
        return x
