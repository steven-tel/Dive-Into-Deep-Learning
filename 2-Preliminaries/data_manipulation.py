import torch


if __name__ == '__main__':

    '''Getting Started'''
    x = torch.arange(12, dtype=torch.float32)
    print(x.numel())
    print(x.shape)
    X = x.reshape(3, 4)
    print(X.shape)
    x = torch.zeros((2, 3, 4))
    x = torch.ones((2, 3, 4))

    # Standard Gaussian distribution with mean 0 and standard deviation 1
    x = torch.randn((2, 3, 4))

    '''Indexing and slicing'''
    x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(x[-1], x[1:3])
    x[1, 1] = 12
    x[0, 0:2] = 9
    print(x)

    '''Operations'''
    print(torch.exp(x))

    x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = torch.tensor([[12, 21, 2], [3, 3, 4], [4, 3, 3]])
    x + y, x - y, x * y, x / y, x ** y

    X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
    Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
    torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)

    print(X == Y)
    print(X.sum())

    '''Broadcasting'''

    a = torch.arange(3).reshape((3, 1))
    b = torch.arange(2).reshape((1, 2))
    print(a + b)

    '''Saving Memory'''
    X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
    y = torch.arange(12, dtype=torch.float32).reshape((3, 4))
    before = id(Y)
    Y = Y + X
    print(id(Y) == before)

    X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
    y = torch.arange(12, dtype=torch.float32).reshape((3, 4))
    before = id(Y)
    Y[:] = Y + X
    print(id(Y) == before)


    '''Conversion to Other Python Objects'''

    A = X.numpy()
    X = torch.from_numpy(A)

    a = torch.tensor([2.4])
    print(a, a.item(), int(a), float(a))
