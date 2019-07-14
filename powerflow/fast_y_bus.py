import os.path
import os
import scipy.sparse
import scipy.io
import numpy as np
from .util.read_data import PowerData
import warnings


def construct_y_bus_data(name, bus_csv_path=None, branch_csv_path=None,
                         force=False):
    # Pass if data has been created already and force is not active
    if not force and check_y_bus_data(name):
        return

    # Check the availablity of the container folder
    check_folder(name)

    # Get the bus data
    if bus_csv_path is None:
        npy_path = os.path.join(os.path.dirname(__file__), 'data', name,
                                'bus_data.npy')
        bus_data = PowerData(name, 'bus', npy_path=npy_path)
    else:
        bus_data = PowerData(name, 'bus', csv_path=bus_csv_path)

    # Get the branch data
    if branch_csv_path is None:
        npy_path = os.path.join(os.path.dirname(__file__), 'data', name,
                                'branch_data.npy')
        branch_data = PowerData(name, 'branch', npy_path=npy_path)
    else:
        branch_data = PowerData(name, 'branch', csv_path=branch_csv_path)

    # y_bus rows and columns labeling by bus_number column,
    # in bus_data_case (in that positioning)
    bus_label = bus_data['bus_number']
    y_bus_size = len(bus_label)

    # Empty y_bus order definition
    y_bus_real = YBusData(name, 'y_bus_real', (y_bus_size, y_bus_size),
                          dtype=np.longdouble)
    y_bus_imag = YBusData(name, 'y_bus_imag', (y_bus_size, y_bus_size),
                          dtype=np.longdouble)

    # y_bus filling process

    # Shunt elements incorporation on bus_data_case:
    # i value on next 'for' is the position of bus_number in bus_label
    # by definition

    shunt_g = scipy.sparse.coo_matrix(bus_data['shunt_g'])
    for i, j, v in zip(shunt_g.row, shunt_g.col, shunt_g.data):
        y_bus_real[j, j] = v

    shunt_b = scipy.sparse.coo_matrix(bus_data['shunt_b'])
    for i, j, v in zip(shunt_b.row, shunt_b.col, shunt_b.data):
        y_bus_imag[j, j] = v

    # Branch elements incorporation on branch_data_case
    for i in range(len(branch_data)):  # For ith element
        # Type of element. 0:transmission line - 1,2,3:transformer
        type_i = branch_data['type'][i]

        # tap_bus; bus_number from one of the buses between the ith element
        # z_bus; bus_number from the other bus

        # tap_bus position detection on y_bus defined in
        # bus_label order
        tap_bus = branch_data['tap_bus_number'][i]
        tap_bus_index = int(np.where(bus_label == tap_bus)[0])

        # z_bus position detection on y_bus defined in
        # bus_label order
        z_bus = branch_data['z_bus_number'][i]
        z_bus_index = int(np.where(bus_label == z_bus)[0])

        branch_data['tap_bus_index'][i] = tap_bus_index
        branch_data['z_bus_index'][i] = z_bus_index

        # Primitive admittance of ith element calculation.
        # Series admittance between tap_bus and z_bus
        # Reactance_i != 0 assumption
        g_li = branch_data['resistance'][i] / \
            (branch_data['resistance'][i] ** 2 +
             branch_data['reactance'][i] ** 2)

        s_li = - branch_data['reactance'][i] / \
            (branch_data['resistance'][i] ** 2 +
             branch_data['reactance'][i] ** 2)

        # Equipment modeling and inclusion to y_bus
        if type_i == 0:  # transmission line, pi model
            g_tap_tap = g_li
            g_z_z = g_li
            g_z_tap = - g_li
            g_tap_z = - g_li

            s_tap_tap = s_li
            s_z_z = s_li
            s_tap_z = - s_li
            s_z_tap = - s_li

            # line charging as shunt capacitor, pi model
            b_li = branch_data['line_charging'][i]
            s_tap_tap += b_li / np.longdouble(2)
            s_z_z += b_li / np.longdouble(2)

        else:  # Transformer (fixed tap or variable tap)
            if branch_data['transf_tr'][i] > 0:
                t = branch_data['transf_tr'][i]
            else:
                t = 1

            t_angle = (branch_data['transf_ang'][i] * np.pi) / \
                np.longdouble(180)
            t_real = t * np.cos(t_angle)
            t_imag = t * np.sin(t_angle)

            g_tap_tap = g_li / (t ** 2)
            g_z_z = g_li

            g_tap_z = - (g_li * t_real - s_li * t_imag) / (t ** 2)
            g_z_tap = - (g_li * t_real + s_li * t_imag) / (t ** 2)

            s_tap_tap = s_li / (t ** 2)
            s_z_z = s_li
            s_tap_z = - (g_li * t_imag + s_li * t_real) / (t ** 2)
            s_z_tap = - (- g_li * t_imag + s_li * t_real) / (t ** 2)

        branch_data.save(os.path.join(os.path.dirname(__file__), 'data', name,
                         'branch_data.npy'))

        # Diagonal elements
        y_bus_real[tap_bus_index, tap_bus_index] += g_tap_tap
        y_bus_imag[tap_bus_index, tap_bus_index] += s_tap_tap

        y_bus_real[z_bus_index, z_bus_index] += g_z_z
        y_bus_imag[z_bus_index, z_bus_index] += s_z_z

        # Non diagonal elements, asymmetric y_bus !
        y_bus_real[tap_bus_index, z_bus_index] += g_tap_z
        y_bus_imag[tap_bus_index, z_bus_index] += s_tap_z

        y_bus_real[z_bus_index, tap_bus_index] += g_z_tap
        y_bus_imag[z_bus_index, tap_bus_index] += s_z_tap

    # Builds y bus magnitude and y bus angle matrices

    # y_bus magnitude and angle construction process
    y_bus_mag = YBusData(name, 'y_bus_mag', (y_bus_size, y_bus_size),
                         dtype=np.longdouble)
    y_bus_angle = YBusData(name, 'y_bus_angle', (y_bus_size, y_bus_size),
                           dtype=np.longdouble)

    values = set()
    coo = y_bus_real.tocoo()
    for i, j in zip(coo.row, coo.col):
        values.add((i, j))
    coo = y_bus_imag.tocoo()
    for i, j in zip(coo.row, coo.col):
        values.add((i, j))

    for index in values:
        # IMPORTANT: An unexpected warning ("division by zero") raises on
        # line 169 (square root calculation), but this operation is not
        # affected. Then, line 168 of this script can be deleted.
        warnings.filterwarnings('ignore')
        y_bus_mag[index] = np.sqrt(y_bus_real[index] ** 2 +
                                   y_bus_imag[index] ** 2)
        y_bus_angle[index] = np.arctan2(y_bus_imag[index], y_bus_real[index])

    # Write the data
    y_bus_real.save()
    y_bus_imag.save()
    y_bus_mag.save()
    y_bus_angle.save()


def check_y_bus_data(name, dir_path=None):
    base_dir = os.path.join(os.path.dirname(__file__), 'data', name)
    if dir_path is not None:
        base_dir = dir_path

    return os.path.exists(os.path.join(base_dir, 'y_bus_angle.mtx')) and \
        os.path.exists(os.path.join(base_dir, 'y_bus_imag.mtx')) and \
        os.path.exists(os.path.join(base_dir, 'y_bus_mag.mtx')) and \
        os.path.exists(os.path.join(base_dir, 'y_bus_real.mtx'))


def check_folder(name):
    if not os.path.isdir(os.path.join(os.path.dirname(__file__), 'data')):
        os.mkdir(os.path.join(os.path.dirname(__file__), 'data'))
    if not os.path.isdir(os.path.join(os.path.dirname(__file__), 'data',
                         name)):
        os.mkdir(os.path.join(os.path.dirname(__file__), 'data', name))


class YBusData(scipy.sparse.lil_matrix):
    def __init__(self, name, filetype, *args, **kwargs):
        super(YBusData, self).__init__(*args, **kwargs)
        self.filetype = filetype
        self.name = name

    def save(self):
        base_path = os.path.join(os.path.dirname(__file__), 'data')
        scipy.io.mmwrite(os.path.join(base_path, self.name, self.filetype +
                         '.mtx'), self.tocsr())

    @staticmethod
    def read(name, filetype, path=None):
        if path is None:
            path = os.path.join(os.path.dirname(__file__), 'data', name,
                                filetype + '.mtx')
        else:
            path = os.path.join(path, filetype + '.mtx')
        return YBusData(name, filetype, scipy.io.mmread(path))


class YBusDataArray(object):
    def __init__(self, name, path=None):
        self.real = YBusData.read(name, 'y_bus_real', path=path)
        self.imag = YBusData.read(name, 'y_bus_imag', path=path)
        self.mag = YBusData.read(name, 'y_bus_mag', path=path)
        self.angle = YBusData.read(name, 'y_bus_angle', path=path)
