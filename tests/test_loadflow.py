import os
import csv
import numpy as np
import powerflow.load_flow


def pytest_generate_tests(metafunc):
    loadflow_settings = []
    for case in ['14', '30', '57', '118', '300']:
        basedir = os.path.join(os.path.dirname(__file__), 'PowerFlow',
                               'LoadFlow', case)
        loadflow_settings.append((case, basedir))
    metafunc.parametrize(('case', 'base_path'), loadflow_settings)


def test_loadflow(case, base_path):
    # IMPORTANT: Only bus_solution values has been tested. line_flows,
    # transformer_flows and shunt_flows not tested yet

    bus_solution, time, iteration, error, line_flows, transformer_flows, \
        shunt_flows, \
        mva_base = powerflow.load_flow.loadflow(case, dir_path=base_path)

    assert iteration < 10
    assert error <= 0.001

    csv_solution = csv.reader(open(os.path.join(base_path, 'solution.csv'),
                              'rU'), delimiter=',')
    i = 0
    for row in list(csv_solution):
        assert float(row[0]) == bus_solution['bus_number'][i]

        # Tension
        assert np.isclose(float(row[1]), bus_solution['v'][i], atol=0.03)

        # Degrees
        assert np.isclose(float(row[2]), bus_solution['delta'][i], atol=2)
        i += 1
