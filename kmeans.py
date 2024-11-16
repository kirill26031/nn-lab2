import torch

# Author: Josue N Rivera (github.com/wzjoriv)
# https://gist.github.com/wzjoriv/7e89afc7f30761022d7747a501260fe3

def random_sample(tensor, k):
    return tensor[torch.randperm(len(tensor))[:k]]

def distance_matrix(x, y=None, p = 2): #pairwise distance of vectors
    
    y = x if type(y) == type(None) else y

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    diff = x - y
    
    dist = torch.linalg.vector_norm(diff, p, 2) if torch.__version__ >= '1.7.0' else torch.pow(diff, p).sum(2)**(1/p)
    
    return dist

class NN():

    def __init__(self, X = None, Y = None, p = 2):
        self.p = p
        self.train(X, Y)

    def train(self, X, Y):
        self.train_pts = X
        self.train_label = Y

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} wasn't trained. Need to execute {name}.train() first")
        
        dist = distance_matrix(x, self.train_pts, self.p)
        labels = torch.argmin(dist, dim=1)
        return self.train_label[labels]


class KMeans(NN):

    def __init__(self, device, X = None, k=2, n_iters = 10, p = 2):

        self.k = k
        self.n_iters = n_iters
        self.p = p
        self.device = device

        if type(X) != type(None):
            self.train(X)

    def train(self, X):

        self.train_pts = random_sample(X, self.k)
        self.train_label = torch.tensor(range(self.k), device=self.device, dtype=torch.int64)

        for _ in range(self.n_iters):
            labels = self.predict(X)

            for lab in range(self.k):
                select = labels == lab
                self.train_pts[lab] = torch.mean(X[select], dim=0)