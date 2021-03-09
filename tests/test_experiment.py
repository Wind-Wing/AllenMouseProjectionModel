import pytest
import numpy as np
from utils.experiment import Experiment
from utils.constants import *


@pytest.fixture(scope='function')
def instantiation():
    shape = [4, 2, 8]
    injection_density = np.zeros(shape)
    projection_density = np.zeros(shape)
    projection_mask = np.zeros(shape)

    injection_density[2][1][1] = 1
    injection_density[3][0][2] = 4
    injection_density[1][1][3] = 2
    injection_density[0][1][5] = 3

    projection_density[2][1][1] = 1
    projection_density[3][0][2] = 4
    projection_density[1][1][3] = 2
    projection_density[0][1][5] = 3
    projection_density[1][1][1] = 2
    projection_density[3][1][7] = 5

    projection_mask[:, :, 4:] = 1
    projection_mask_idx = projection_mask.nonzero()

    exp = Experiment(None, projection_mask_idx, None, injection_density, projection_density)

    return exp


@pytest.fixture(scope='function')
def right_instantiation():
    shape = VOXEL_SHAPE
    injection_density = np.zeros(shape)
    projection_density = np.zeros(shape)
    projection_mask = np.zeros(shape)

    injection_density[60][20][70] = 1
    projection_density[60][20][70] = 1
    projection_density[50][30][30] = 2
    projection_density[60][20][VOXEL_SHAPE[2] - 1 - 70] = 3
    projection_mask[60][20][70] = 1

    projection_mask_idx = projection_mask.nonzero()
    exp = Experiment(None, projection_mask_idx, R_HEMISPHERE, injection_density, projection_density)
    return exp


@pytest.fixture(scope='function')
def left_instantiation():
    shape = VOXEL_SHAPE
    injection_density = np.zeros(shape)
    projection_density = np.zeros(shape)
    projection_mask = np.zeros(shape)

    injection_density[60][20][30] = 1

    projection_density[60][20][30] = 1
    projection_density[60][20][VOXEL_SHAPE[2] - 1 - 30] = 3
    projection_density[50][30][30] = 2

    projection_mask[60][20][30] = 1
    projection_mask[60][20][VOXEL_SHAPE[2] - 1 - 30] = 1
    projection_mask[50][30][VOXEL_SHAPE[2] - 1 - 30] = 1

    projection_mask_idx = projection_mask.nonzero()
    exp = Experiment(None, projection_mask_idx, R_HEMISPHERE, injection_density, projection_density)
    return exp


def test_instantiation(instantiation):
    exp = instantiation
    assert list(exp.injection_centroid) == [1.6, 0.6, 3]

    expected_value = [0] * (4*2*4)
    expected_value[5] = 0.3
    expected_value[-1] = 0.5
    assert list(exp.normalized_projection_density) == expected_value


def test_right_instantiation(right_instantiation):
    exp = right_instantiation
    assert exp.hemisphere == R_HEMISPHERE
    assert list(exp.injection_centroid) == [60, 20, 70]

    expected_value = [1]
    assert list(exp.normalized_projection_density) == expected_value


def test_left_instantiation(left_instantiation):
    exp = left_instantiation
    assert exp.hemisphere == L_HEMISPHERE
    assert list(exp.injection_centroid) == [60, 20, VOXEL_SHAPE[2] - 1 - 30]

    expected_value = [2, 3, 1]
    assert list(exp.normalized_projection_density) == expected_value




