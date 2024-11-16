import torch

# Author: Josue N Rivera (github.com/wzjoriv)
# https://gist.github.com/wzjoriv/7e89afc7f30761022d7747a501260fe3

def random_sample(tensor, k):
    return tensor[torch.randperm(len(tensor))[:k]].detach().clone()

def distance_matrix(x, y=None, p = 2): #pairwise distance of vectors
    
    y = x if type(y) == type(None) else y

    n = x.size(0)
    m = y.size(0)
    
    dist = torch.empty(n, m, device=x.device)  # Allocate memory for result

    # Compute distances in a memory-efficient way
    for i in range(m):  # Loop over y vectors
        diff = torch.sub(x, y[i])   # Broadcast only over x, avoiding full tensor expansion
        dist[:, i] = torch.linalg.vector_norm(diff, p, dim=1) if torch.__version__ >= '1.7.0' else torch.pow(diff, p).sum(1).pow(1/p)

    return dist

class NN():

    def __init__(self, X = None, Y = None, p = 2):
        self.p = p
        self.train(X, Y)

    def train(self, X, Y):
        self.train_pts = X
        self.train_label = Y.to(torch.int32)

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


class KNN(NN):

    def __init__(self, X = None, Y = None, k = 3, p = 2):
        self.k = k
        super().__init__(X, Y, p)
    
    def train(self, X, Y):
        super().train(X, Y)
        if type(Y) != type(None):
            self.unique_labels = self.train_label.unique()

    def predict(self, x):
        if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} wasn't trained. Need to execute {name}.train() first")
        
        dist = distance_matrix(x, self.train_pts, self.p)

        knn = dist.topk(self.k, largest=False)
        votes = self.train_label[knn.indices]

        winner = torch.zeros(votes.size(0), dtype=votes.dtype, device=votes.device)
        count = torch.zeros(votes.size(0), dtype=votes.dtype, device=votes.device) - 1

        for lab in self.unique_labels:
            vote_count = (votes == lab).sum(1).to(dtype=torch.int32)
            who = vote_count >= count
            winner[who] = lab
            count[who] = vote_count[who]

        return winner