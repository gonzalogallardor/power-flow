import csv
import os.path
import numpy as np


class PowerData(object):

    # Class Definitions

    def __init__(self, name, type, csv_path=None, npy_path=None):
        """
        Constructor of the PowerData class
        It creates a representation of the matrix based on csv with the utility
        of text index acces like a dictionary
        """

        assert type in ['bus', 'branch'], 'Type not recognized, try ["bus",' \
            'data"]'
        assert csv_path is not None or npy_path is not None, 'Needs to be a' \
            'source to instantiate'

        if npy_path is not None:
            self.read(npy_path)
        else:
            self.construct(csv_path, type)
            self.save(os.path.join(os.path.dirname(__file__), '..', 'data',
                      name, 'bus_data.npy' if type == 'bus'
                      else 'branch_data.npy'))

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)

    def construct(self, data_path, type):
        reader = csv.reader(open(data_path, "rU"), delimiter=',')
        # self.data = np.array( [ ( np.longdouble(column) for column in row )
        # for row in list(reader) ], dtype=[] )
        data = []
        for row in list(reader):
            obj = []
            for column in row:
                obj.append(column)
            if type == 'branch':
                obj.extend([0, 0])
            data.append(tuple(obj))

        if type == 'bus':
            self.data = np.array(data, dtype=[
                ('bus_number', 'i'),   # Bus number
                ('area_number', 'i'),  # Region or company number
                ('loss_number', 'i'),  # Loss zone number
                ('type', 'i'),         # Bus type [
                                       #   0: PQ bus without charge,
                                       #   1: PQ bus with MVAR generation and
                                       #       acoted tension,
                                       #   2: PV bus with MVAR limit,
                                       #   3: Slack bus. It must exist.
                                       #       [Unique]
                ('v', 'f8'),           # Initial tension. On PV and slack is
                                       # fixed
                ('delta', 'f8'),       # Initial angle. Fixed on slack bus
                ('pd', 'f8'),          # Active demanding power. Only on PQ
                                       # bus
                ('qd', 'f8'),          # Demand inductive reactive power.
                                       # Only on PQ bus
                ('pg', 'f8'),          # Generated active power. Only on PV
                                       # and eventualy on PQ bus
                ('qg', 'f8'),          # Generated inductive reactive power.
                                       # Only on PQ bus
                ('base_kv', 'f8'),     # Base tension of the bus
                ('desired_v', 'f8'),   # Desired Tension
                ('max_limit', 'f8'),   # Max reactive inductive power
                ('min_limit', 'f8'),   # Min reactive inductive power
                ('shunt_g', 'f8'),     # shunt Conductance
                ('shunt_b', 'f8'),     # shunt Susceptance
                ('remote_i', 'i')      # bus_number of the bus remotely
                                       # controlled by
                ])

        else:
            self.data = np.array(data, dtype=[
                ('tap_bus_number', 'i'),       # Bus number in one extreme
                                               # (associated to tap)
                ('z_bus_number', 'i'),         # Bus number on the other
                                               # extreme (associated)
                ('area_number', 'i'),          # Region or company number
                ('loss_number', 'i'),          # Loss zone number
                ('circuit', 'i'),              # Number of transforms/paralel
                                               # lines
                ('type', 'i'),                 # Element type [
                                               #   0: Transmision line,
                                               #   1: Fixed tap transform,
                                               #   2: Variable tap transform
                                               #      for tension control,
                                               #   3: Variable tap transform
                                               #      for MVAR control,
                                               #   4: Variable tap transform
                                               #       for MW control]
                ('resistance', 'f8'),          # Resistance
                ('reactance', 'f8'),           # Reactance
                ('line_charging', 'f8'),       # Load current line
                ('mva_rating1', 'f8'),         # MVA Charge type 1
                ('mva_rating2', 'f8'),         # MVA Charge type 2
                ('mva_rating3', 'f8'),         # MVA Charge type 3
                ('control_bus_number', 'i'),   # Bus number for the bus
                                               # controlled by
                ('side', 'i'),                 # Relative orientation [
                                               #   0: Controlled bus at any
                                               #   side,
                                               #   1: Controlled bus at taps
                                               #      side,
                                               #   2: Controlled bus at no
                                               #      taps side]
                ('transf_tr', 'f8'),           # Actual number of turns in the
                                               #  transform
                ('transf_ang', 'f8'),          # Gap betwen tension fasors
                ('transf_mintap', 'f8'),       # State of the transform or min
                                               #  tab position
                ('transf_maxtap', 'f8'),       # State of the transform or max
                                               # tap position
                ('step_size', 'f8'),           # Size of the tap gap
                ('min_limit', 'f8'),           # Min tension limit
                ('max_limit', 'f8'),           # Max tension limit
                ('tap_state', 'f8'),           # Tap state
                ('tap_bus_index', 'i'),
                ('z_bus_index', 'i')
                ])

    def save(self, data_path):
        np.save(data_path, self.data)

    def read(self, data_path):
        self.data = np.load(data_path)

    # Static Definitions

    @staticmethod
    def get_mva_base():
        # IMPORTANT: MVA_BASE MUST BE AN ARGUMENT FROM PER UNIT ALGORITHM,
        # NOT A DEFAULT VALUE
        return np.longdouble(100)
