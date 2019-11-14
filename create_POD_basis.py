import numpy as np
import csv_parser as cp


def calculate_POD_basis(snapshots):
    '''
    :param snapshots: snapshots imported as 2-D array of size (# of elements, # of snapshots)
    :return:
    '''
    n_elmts = np.shape(snapshots)[0]
    n_snapshots = np.shape(snapshots)[1]
    # Calculate covariance matrix
    sum_UUT = np.matrix(np.zeros((n_elmts, n_elmts)))
    for t in range(n_snapshots):
        U = snapshots[:, t]
        U = np.reshape(U, (n_elmts, 1))
        UUT = np.matmul(U, U.T)
        sum_UUT = sum_UUT + UUT
    R = 1/n_snapshots * sum_UUT
    K = R
    D, V = np.linalg.eig(K)
    V = V.real
    D = D.real
    I = np.argsort(D)[::-1]
    D = D[I]
    V = V[:, I]
    return create_basis(V, D, energy=0.99)


def create_basis(V, D, energy):
    '''
    Creates basis functions for POD using SVD decomposition
    :param V: eigenvectors of covariance matrix
    :param D: diagonal matrix with eigenvalues of covariance matrix
    :param energy: parameter for how much "energy" is desired to be capture in POD basis (usually 95%+)
    :return: POD basis phi
    '''
    cum_sum = 0
    total_sum = sum(D).real
    m = 0  # Calculate number of POD bases that will be needed
    for i in range(0, len(D)):
        cum_sum += D[i].real
        percentage_energy = cum_sum/total_sum
        if percentage_energy < energy:
            m = i
        else:
            m = i
            break
    # Create new matrix based on POD values at each of the sub domains given by value in regionDictionary
    # phi is (n, m) for n = number of elements in mesh, m = number of elements in POD
    phi = V[:, 0:m + 1]
    return phi.real


if __name__ == '__main__':
    # Animating a single agent, assuming agents live on a 2D plane
    pos_arr = cp.parse_csv('swarm_ring_500_a1.csv')
    pos_arr = np.array(pos_arr)
    x_coor = np.zeros([pos_arr.shape[0],pos_arr.shape[1]])
    y_coor = np.zeros([pos_arr.shape[0],pos_arr.shape[1]])
    for i in range(len(pos_arr)):
        for j in range(len(pos_arr[i])):
            x_coor[i][j] = pos_arr[i][j][0]
            y_coor[i][j] = pos_arr[i][j][0]
    basis_x = calculate_POD_basis(x_coor)
    basis_y = calculate_POD_basis(y_coor)
    print basis_y.shape