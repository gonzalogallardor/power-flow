#!/usr/bin/env python

from powerflow.util.read_data import PowerData
from scipy.io import mmread
import powerflow.fast_y_bus
import powerflow.load_flow
from powerflow.exceptions import YBusDataTypeError
import os


class PowerFlow:
    """
    This class stores bus and branch data and paths to save and read
    y_bus_real, y_bus_imag, y_bus_mag and y_bus_angle matrices
    """

    def __init__(self, instance_name, bus_path=None, branch_path=None,
                 force=False):

        """
        Initialize a Powerflow instance for a given case
        """
        if bus_path and branch_path:

            data_path = os.path.join(os.path.dirname(__file__), 'data',
                                     instance_name)

            if not os.path.isfile(bus_path) and \
                    not os.path.isfile(branch_path):
                raise YBusDataTypeError('Invalid path to bus and branch data')

            if not os.path.isfile(bus_path):
                raise YBusDataTypeError('Invalid path to bus data')

            if not os.path.isfile(branch_path):
                raise YBusDataTypeError('Invalid path to branch data')

            if os.path.isdir(data_path) and not force:
                raise YBusDataTypeError('Instance "' +
                                        instance_name + '" '
                                        'already exists')

            powerflow.fast_y_bus.check_folder(instance_name)

            self.mva_base = PowerData.get_mva_base()
            self.bus_data = PowerData(instance_name, 'bus', csv_path=bus_path)
            self.branch_data = PowerData(instance_name, 'branch',
                                         csv_path=branch_path)

            self.y_bus_real_path = os.path.join(data_path, 'y_bus_angle.mtx')
            self.y_bus_imag_path = os.path.join(data_path, 'y_bus_imag.mtx')
            self.y_bus_mag_path = os.path.join(data_path, 'y_bus_mag.mtx')
            self.y_bus_angle_path = os.path.join(data_path, 'y_bus_real.mtx')

            if force or (not force and
                         not os.path.exists(self.y_bus_real_path)):

                powerflow.fast_y_bus.construct_y_bus_data(instance_name,
                                                          bus_path,
                                                          branch_path,
                                                          force)

        else:
            self.case = instance_name
            self.bus_solution = None
            self.iteration = None
            self.error = None
            self.time = None
            self.line_flows = None

    def get_y_bus_real(self):
        y_bus_real = mmread(self.y_bus_real_path)
        return y_bus_real

    def get_y_bus_imag(self):
        y_bus_imag = mmread(self.y_bus_imag_path)
        return y_bus_imag

    def get_y_bus_mag(self):
        y_bus_mag = mmread(self.y_bus_mag_path)
        return y_bus_mag

    def get_y_bus_angle(self):
        y_bus_angle = mmread(self.y_bus_angle_path)
        return y_bus_angle

    def load_flow(self, max_iter=10, tol=0.01, decimals=4,
                  reactive_limits=False, tap_v_control=False,
                  tap_q_control=False, dir_path=None):
        """
        Main function to load the metadata and iterates throw the proccess
        to form the power flow
        """

        self.bus_solution, self.time, \
            self.iteration, self.error, \
            self.line_flows, self.transf_flows, \
            self.shunt_flows, self.mva_base = powerflow.load_flow \
            .loadflow(self.case,
                      max_iter=max_iter,
                      tol=tol,
                      decimals=decimals,
                      reactive_limits=reactive_limits,
                      tap_v_control=tap_v_control,
                      tap_q_control=tap_q_control,
                      dir_path=dir_path)

    def get_load_flow_solution(self):
        assert self.bus_solution is not None, 'The load flow operation is' \
            'not complete'
        return self.bus_solution, self.time, self.iteration, self.error, \
            self.line_flows, self.transf_flows, self.shunt_flows, self.mva_base
