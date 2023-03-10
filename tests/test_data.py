from tests import _PATH_DATA
from src.data.make_dataset import CorruptMnist
import pytest


global data_train
data_train = CorruptMnist(train=True)
global data_test
data_test = CorruptMnist(train=False)

# @pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_data_train():
    assert len(data_train) == 25000 # for training and N_test for test
    # assert that each datapoint has shape [1,28,28] or [728] depending on how you choose to format
    # assert that all labels are represented


def test_data_test():
    assert len(data_test) == 5000

@pytest.mark.skipif(reason="no way of testing nothingness")
def test_nothing():
    assert True
    # print("nothing here")

@pytest.mark.parametrize("test_input,expected", [("3+5", 8), ("2+4", 6), ("6+9", 15)])
def test_eval(test_input, expected):
    assert eval(test_input) == expected
