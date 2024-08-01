
from nb_functools import nb_functools


def test_calc_m1v1_m1v2():

    result = nb_functools.calc_m1v1_m2v2(m1=10000, v1=None, m2=50, v2=200)
    assert result == 1
