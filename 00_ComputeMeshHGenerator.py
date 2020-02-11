import pyigl as igl
from homology import *
from iglhelpers import *
import meshplot as mp

def faceToEdges(f):
    signs = []

    e1 = (f[0], f[1])
    if e1[0] > e1[1]:
        e1 = (e1[1], e1[0])
        signs.append(-1)
    else:
        signs.append(1)

    e2 = (f[1], f[2])
    if e2[0] > e2[1]:
        e2 = (e2[1], e2[0])
        signs.append(-1)
    else:
        signs.append(1)

    e3 = (f[2], f[0])
    if e3[0] > e3[1]:
        e3 = (e3[1], e3[0])
        signs.append(-1)
    else:
        signs.append(1)

    edges = [e1, e2, e3]
    return edges, signs


class Mesh:
    def __init__(self, verts, faces):
        self.vertsMat = verts
        self.facesMat = faces

        self.faceSet = set()
        self.edgeIdMap = {}

        numE = 0
        for f in faces:
            self.faceSet.add(tuple(f.tolist()))
            edges, signs = faceToEdges(f)
            for e in edges:
                if e not in self.edgeIdMap:
                    self.edgeIdMap[e] = numE
                    numE +=1

        self.nV = verts.shape[0]
        self.nE = len(self.edgeIdMap)
        self.nF = len(self.faceSet)

    def getD2(self):
        d2 = np.zeros((self.nE, self.nF))
        for iF, f in enumerate(self.facesMat.tolist()):
            edges, signs = faceToEdges(f)
            for e, sign in zip(edges, signs):
                eId = self.edgeIdMap[e]
                d2[eId, iF] = sign

        return d2

    def getD1(self):
        d1 = np.zeros((self.nV, self.nE))
        for e, eId in self.edgeIdMap.items():
            v1 = e[0]
            v2 = e[1]
            d1[v1, eId] = -1
            d1[v2, eId] = 1
        return d1




if __name__ == '__main__':
    V = igl.eigen.MatrixXd()
    F = igl.eigen.MatrixXi()
    # igl.readOBJ('Circle.obj', V, F)
    igl.readOBJ('TorusSimplified.obj', V, F)

    verts = e2p(V)
    faces = e2p(F)

    mp.plot(verts, faces)

    print(verts.shape)
    print(faces.shape)

    mesh = Mesh(verts, faces)

    print(mesh.edgeIdMap)

    d2 = mesh.getD2()
    print('d2:\n', d2)

    d1 = mesh.getD1()
    print('d1:\n', d1)

    A = d1.astype(numpy.float64)
    B = d2.astype(numpy.float64)

    A, B, Q, QInv = simultaneousReduceWithQ(A, B)
    # B = finishRowReducing(B)

    print('A', A)
    print('B', B)

    print("Q:\n", Q)
    print("QInv:\n", QInv)

    # get generator for the homology group
    z = numpy.zeros(A.shape[0])
    basisKer = [i for i in range(A.shape[1]) if numpy.all(A[:, i] == z)]

    firstZeroCol = -1
    for i in range(A.shape[1])[::-1]:
        if  numpy.all(A[:, i] == z):
            firstZeroCol = i
        else:
            break

    # z = numpy.zeros(B.shape[1])
    # basisIm = [i for i in range(B.shape[0]) if numpy.any(B[i, :] != z)]
    basisIm = []
    for i in range(B.shape[0] - firstZeroCol):
        if i + firstZeroCol >= B.shape[0] or i >= B.shape[1]:
            break

        if B[i+firstZeroCol, i]:
            basisIm.append(i+firstZeroCol)


    basisH = numpy.setdiff1d(basisKer, basisIm)

    idToEMap = {eId:e for e, eId in mesh.edgeIdMap.items()}
    print("Generator for homology group:")
    for i in basisH:
        print(Q[:, i])
        eIds = np.where(Q[:, i])[0]
        for eId in eIds:
            print("" if Q[eId, i] > 0 else "-",  idToEMap[eId])
