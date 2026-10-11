import numpy as np
import numpy.typing as npt

hidden_size = 8

def inputs_of_controller(data, target_position): 
    target_xy=np.array(target_position[:2],dtype=float)
    current_xy=np.array(data.qpos[:2],dtype=float)
    data_2d=np.array(data.qpos,dtype=float)
    direction = target_xy - current_xy
    phase=2 * np.pi * data.time
    return np.concatenate((data_2d, direction, [np.sin(phase), np.cos(phase)]))

def genotype_length(N: int, hidden_size: int, M: int) -> int:
    return N * hidden_size + hidden_size * M

def genotype_to_weights(genotype: npt.NDArray[np.float64], N: int, hidden_size: int, M: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:

    if genotype.shape[0] != genotype_length(N, hidden_size, M):
        raise ValueError("Invalid genotype length")
    point=N*hidden_size
    W1 = genotype[:point].reshape(N, hidden_size)
    W2 = genotype[point:].reshape(hidden_size, M)
    return W1, W2

def angle_of_controller(inputs: npt.NDArray[np.float64], W1: npt.NDArray[np.float64], W2: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    hidden = np.tanh(np.dot(inputs, W1))
    output = np.tanh(np.dot(hidden, W2))
    return output * (np.pi / 2.0)
