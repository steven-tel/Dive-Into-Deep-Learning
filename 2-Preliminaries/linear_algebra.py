import torch


'''https://d2l.ai/chapter_preliminaries/linear-algebra.html'''


if __name__ == '__main__':

    '''Scalars'''
    x = torch.tensor(3.0)
    y = torch.tensor(2.0)

    print(x+y, x * y,  x / y, x**y)

    '''Vectors'''
    x = torch.arange(3)
    print(x, x[2])
    print(len(x), x.shape)

    '''Matrices'''
    x = torch.arange(12).reshape(3, 4)
    print(x)
    print(x, x.T)
    print(x.shape, x.T.shape)

    '''Tensors'''
    x = torch.arange(27).reshape(3, 3, 3)
    print(x[1, 1, 1])

    '''Basic Propeties of Tensor Arithmetic'''
    A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    B = A.clone()
    print(A, A + B, A * B)

    a = 2
    X = torch.arange(24).reshape(2, 3, 4)
    a + X, (a * X).shape

    '''Reduction'''
    x = torch.arange(3, dtype=torch.float32)
    print(x, x.sum())
    print(A.shape, A.sum(axis=0).shape)
    print(A.mean(axis=0), A.sum(axis=0) / A.shape[0])

    '''Non-Reduction Sum'''
    sum_A = A.sum(axis=1, keepdims=True)
    print(A, sum_A)
    print(A / sum_A)
    print(A, A.cumsum(axis=0))

    '''Dot Products'''
    y = torch.ones(3, dtype=torch.float32)
    print(torch.dot(x, y))

    '''Matrix-Vector Products'''
    print(A.shape, x.shape, torch.mv(A, x), A@x)

    '''Matrix-Matrix Multiplication'''
    B = torch.ones(3, 4)
    B = torch.ones(3, 4)
    print(torch.mm(A, B), A @ B)

    '''Norms'''
    u = torch.tensor([3.0, -4.0])
    # L2
    print(torch.norm(u))
    # L1
    print(torch.abs(u).sum())
    # Frobenius norm
    print(torch.norm(torch.ones((4, 9))))

