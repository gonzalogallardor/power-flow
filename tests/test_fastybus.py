import pdb
import numpy as np
import os
import powerflow.fast_y_bus
import powerflow.load_flow


def pytest_generate_tests(metafunc):
    """
    Generate a list of strings where each element
    is a bus power flow case
    """

    fastybus_settings = []
    for case in ['14', '30', '57', '118', '300']:
        basedir = os.path.join(os.path.dirname(__file__), 'PowerFlow',
                               'FastYBus', case)
        fastybus_settings.append((case, basedir))
    metafunc.parametrize(('case', 'basedir'), fastybus_settings)


def test_fastybus(case, basedir):
    """
    Check content of y_bus_real, y_bus_imag, y_bus_mag and y_bus_angle
    for a given bus power flow case
    """

    bus_path = os.path.join(basedir, 'bus_data.csv')
    branch_path = os.path.join(basedir, 'branch_data.csv')

    powerflow.fast_y_bus.construct_y_bus_data(case, bus_csv_path=bus_path,
                                              branch_csv_path=branch_path,
                                              force=True)

    # Check the integrity of the y_bus_imag
    imag = powerflow.fast_y_bus.YBusData.read(case, 'y_bus_imag')
    assert np.linalg.matrix_rank(imag.todense()) == (imag.shape[0])
