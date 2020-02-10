from homology import *

if __name__ == '__main__':
    # bd0 = numpy.array([[0, 0, 0, 0, 0]])
    # bd1 = numpy.array([[-1, -1, -1, -1, 0, 0, 0, 0], [1, 0, 0, 0, -1, -1, 0, 0],
    #                    [0, 1, 0, 0, 1, 0, -1, -1], [0, 0, 1, 0, 0, 1, 1, 0], [0, 0, 0, 1, 0, 0, 0, 1]])
    # bd2 = numpy.array([[1, 1, 0, 0], [-1, 0, 1, 0], [0, -1, -1, 0],
    #                    [0, 0, 0, 0], [1, 0, 0, 1], [0, 1, 0, -1],
    #                    [0, 0, 1, 1], [0, 0, 0, 0]])
    # bd3 = numpy.array([[-1], [1], [-1], [1]])
    bd1 = numpy.array([
        [-1,-1,-1,0,0,0,0],
        [1,0,0,-1,0,0,0],
        [0,1,0,1,-1,-1,0],
        [0,0,1,0,1,0,-1],
        [0,0,0,0,0,1,1]
        ])

    bd2 = numpy.array([
      [0],
        [1],
        [-1],
        [0],
        [1],
        [0],
        [0],
    ])

    A = bd1.astype(numpy.float64)
    B = bd2.astype(numpy.float64)

    A, B, Q, QInv = simultaneousReduceWithQ(A, B)

    print(A)
    print(B)

    print("Q:\n", Q)
    print("QInv:\n", QInv)

    # get generator for the homology group
    z = numpy.zeros(B.shape[1])
    basisIm = [i for i in range(B.shape[0]) if numpy.any(B[i, :] != z)]

    z = numpy.zeros(A.shape[0])
    basisKer = [i for i in range(A.shape[1]) if numpy.all(A[:, i] == z)]

    basisH = numpy.setdiff1d(basisKer, basisIm)

    print("Generator for homology group:")
    for i in basisH:
        print(Q[:, i])

    # finishRowReducing(B)
    # print(B)