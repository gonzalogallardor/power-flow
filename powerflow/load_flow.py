import os.path
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from powerflow import fast_y_bus
from powerflow.util import read_data
from powerflow.exceptions import LoadFlowDataError
import copy
import csv
import datetime


def loadflow(case, max_iter=10, tol=0.01, decimals=4, reactive_limits=False,
             tap_v_control=False, tap_q_control=False, dir_path=None):

    start_time = datetime.datetime.now()

    if not fast_y_bus.check_y_bus_data(case, dir_path):
        raise LoadFlowDataError("The data for the case: '" + case + "' "
                                "doesn't exist")

    # Get the bus and the y bus data and set paths to store solutions and
    # general data

    if dir_path is None:
        bus_npy_path = os.path.join(os.path.dirname(__file__), 'data', case,
                                    'bus_data.npy')

        branch_npy_path = os.path.join(os.path.dirname(__file__), 'data',
                                       case, 'branch_data.npy')

        bus_data = read_data.PowerData(case, 'bus', npy_path=bus_npy_path)

        branch_data = read_data.PowerData(case, 'bus',
                                          npy_path=branch_npy_path)

        bus_solution_path = os.path.join(os.path.dirname(__file__), 'data',
                                         case, 'bus_solution.csv')

        line_flows_path = os.path.join(os.path.dirname(__file__), 'data',
                                       case, 'line_flows.csv')

        transformer_flows_path = os.path.join(os.path.dirname(__file__),
                                              'data', case,
                                              'transformer_flows.csv')

        shunt_flows_path = os.path.join(os.path.dirname(__file__), 'data',
                                        case, 'shunt_flows.csv')

        general_data_path = os.path.join(os.path.dirname(__file__), 'data',
                                         case, 'general_data.csv')

    else:
        bus_data = read_data.PowerData(case, 'bus',
                                       npy_path=os.path.join(dir_path,
                                                             'bus_data'
                                                             '.npy'))

        branch_data = read_data.PowerData(case, 'branch',
                                          npy_path=os.path.join(dir_path,
                                                                'branch_data'
                                                                '.npy'))

        bus_solution_path = os.path.join(dir_path, 'bus_solution.csv')

        line_flows_path = os.path.join(dir_path, 'line_flows.csv')

        transformer_flows_path = os.path.join(dir_path,
                                              'transformer_flows.csv')
        shunt_flows_path = os.path.join(dir_path, 'shunt_flows.csv')

        general_data_path = os.path.join(dir_path, 'general_data.csv')

    y_bus = fast_y_bus.YBusDataArray(case, path=dir_path)

    # Get the mva base
    mva_base = read_data.PowerData.get_mva_base()

    # Create the pv_data, pq_data and slack_data
    # Ordered information from bus_data and mva base
    data = LoadFlowData(bus_data, mva_base)
    data_ini = copy.deepcopy(data)

    # Main Loop of Load Flow
    for iteration in range(max_iter):
        # p and q iteration calc
        p_calc, q_calc = calc_iter(data, y_bus)

        # delta_p, delta_q calc
        # delta_p, delta_q are addittional information not used
        full_delta_p, full_delta_q = delta_p_q(data, p_calc, q_calc)

        # Cut information and concatenate delta_p, delta_q
        delta_pq = cut_delta_p_q(data, full_delta_p, full_delta_q)
        error = max(delta_pq)

        # Error comparison
        if tol/mva_base > np.abs(error):
            break

        # Construct the Jacobian only on even iterations
        m_mtx, n_mtx = m_n_matrix(data, y_bus, p_calc, q_calc)
        j_inv = j_construction(data, y_bus, m_mtx, n_mtx)

        delta_sv = j_inv.dot(delta_pq)

        sv_update(data, delta_sv)

        if reactive_limits and iteration > 0 and iteration % 3 == 0:
            if iteration == 2:
                changed_buses = []
                changed_voltages = []

            changed_buses, changed_voltages= q_limit_consider(data, 
                                                              data_ini, 
                                                              mva_base, 
                                                              iteration,
                                                              changed_buses,
                                                              changed_voltages)

        if tap_v_control and iteration > 0 and iteration % 3 == 0:
            target_tol = 5  # percentage of tolerance admitted for target voltage = (min_limit + max_limit) / 2
            tol_tr = 10  # percentage of admitted turns_ratio variation
            transf_data, action, controled_buses = v_control_data(data, 
                                                                  target_tol / 100, 
                                                                  branch_data, 
                                                                  iteration)
            if iteration == 0:
                print('Warnining: Following buses are voltage controled by '
                      'tap transformers (bus_number):')
                print(controled_buses)
    
            transf_data = v_control_sensitivity(data, 
                                                j_inv, 
                                                transf_data, 
                                                action)
            v_control_tap_update(data, 
                                 y_bus, 
                                 branch_data, 
                                 transf_data, 
                                 tol_tr)
        
        print('----- Iteration: ', iteration + 1)
        print(error)

    time = datetime.datetime.now() - start_time

    iteration += 1

    bus_solution = sol_bus_solution(data)
    line_flows = sol_line_flows(bus_solution, bus_data, branch_data, mva_base)
    transformer_flows = sol_transformer_flows(bus_solution, bus_data,
                                              branch_data, mva_base)
    shunt_flows = sol_shunt_flows(bus_solution, bus_data, mva_base)

    # Round bus_solution and line_flows elements to the desired number of
    # decimals
    round_solution(bus_solution, decimals)
    round_solution(line_flows, decimals)
    round_solution(transformer_flows, decimals)
    round_solution(shunt_flows, decimals)

    # Write bus_solution and line_flows to a csv file each one
    write_solution(bus_solution, bus_solution_path)
    write_solution(line_flows, line_flows_path)
    write_solution(transformer_flows, transformer_flows_path)
    write_solution(shunt_flows, shunt_flows_path)

    # Write time, iteration and error to a csv file
    write_general_data(mva_base, time, iteration, error, general_data_path)

    return bus_solution, time, iteration, error, line_flows, \
        transformer_flows, shunt_flows, mva_base

def calc_iter(data, y_bus):

    # PQ and PV loop
    for i in range(max(data.pq.shape[0], data.pv.shape[0])):
        # Slack data
        if i == 0:

            data.slack['p_calc'] = data.slack['v'] ** 2 \
                * y_bus.real[data.slack['bus_index'], data.slack['bus_index']]

            data.slack['p_calc'] = data.slack['v'] ** 2 \
                * y_bus.imag[data.slack['bus_index'], data.slack['bus_index']]

        # PQ data
        if i < data.pq.shape[0]:

            data.pq['p_calc'][i] = data.pq['v'][i] ** 2 \
                * y_bus.real[data.pq['bus_index'][i],
                             data.pq['bus_index'][i]] \
                + data.pq['v'][i] \
                * data.slack['v'] \
                * y_bus.mag[data.pq['bus_index'][i],
                            data.slack['bus_index']] \
                * np.cos(
                         y_bus.angle[data.pq['bus_index'][i],
                                     data.slack['bus_index']] +
                         data.slack['delta'] -
                         data.pq['delta'][i])

            data.pq['q_calc'][i] = data.pq['v'][i] ** 2 \
                * y_bus.imag[data.pq['bus_index'][i],
                             data.pq['bus_index'][i]] \
                + data.pq['v'][i] \
                * data.slack['v'] \
                * y_bus.mag[data.pq['bus_index'][i],
                            data.slack['bus_index']] \
                * np.sin(
                        y_bus.angle[data.pq['bus_index'][i],
                                    data.slack['bus_index']] +
                        data.slack['delta'] -
                        data.pq['delta'][i])

        # PV data
        if i < data.pv.shape[0]:
            data.pv['p_calc'][i] = data.pv['v'][i] ** 2 \
                * y_bus.real[data.pv['bus_index'][i],
                             data.pv['bus_index'][i]] \
                + data.pv['v'][i] \
                * data.slack['v'] \
                * y_bus.mag[data.pv['bus_index'][i],
                            data.slack['bus_index']] \
                * np.cos(
                         y_bus.angle[data.pv['bus_index'][i],
                                     data.slack['bus_index']] +
                         data.slack['delta'] -
                         data.pv['delta'][i])

            data.pv['q_calc'][i] = data.pv['v'][i] ** 2 \
                * y_bus.imag[data.pv['bus_index'][i],
                             data.pv['bus_index'][i]] \
                + data.pv['v'][i] \
                * data.slack['v'] \
                * y_bus.mag[data.pv['bus_index'][i],
                            data.slack['bus_index']] \
                * np.sin(
                         y_bus.angle[data.pv['bus_index'][i],
                                     data.slack['bus_index']] +
                         data.slack['delta'] -
                         data.pv['delta'][i])

        # PQ and PV sub-loop
        for j in range(max(data.pq.shape[0], data.pv.shape[0])):
            # PQ bus
            if j < data.pq.shape[0]:
                # Slack data
                if i == 0:
                    data.slack['p_calc'] += data.slack['v'] \
                        * data.pq['v'][j] \
                        * y_bus.mag[data.slack['bus_index'],
                                    data.pq['bus_index'][j]] \
                        * np.cos(
                                 y_bus.angle[data.slack['bus_index'],
                                             data.pq['bus_index'][j]] +
                                 data.pq['delta'][j] -
                                 data.slack['delta'])

                    data.slack['q_calc'] += data.slack['v'] \
                        * data.pq['v'][j] \
                        * y_bus.mag[data.slack['bus_index'],
                                    data.pq['bus_index'][j]] \
                        * np.sin(
                                 y_bus.angle[data.slack['bus_index'],
                                             data.pq['bus_index'][j]] +
                                 data.pq['delta'][j] -
                                 data.slack['delta'])

                # PQ data
                if i < data.pq.shape[0] and i != j:
                    data.pq['p_calc'][i] += data.pq['v'][i] \
                        * data.pq['v'][j] \
                        * y_bus.mag[data.pq['bus_index'][i],
                                    data.pq['bus_index'][j]] \
                        * np.cos(
                                 y_bus.angle[data.pq['bus_index'][i],
                                             data.pq['bus_index'][j]] +
                                 data.pq['delta'][j] -
                                 data.pq['delta'][i])

                    data.pq['q_calc'][i] += data.pq['v'][i] \
                        * data.pq['v'][j] \
                        * y_bus.mag[data.pq['bus_index'][i],
                                    data.pq['bus_index'][j]] \
                        * np.sin(
                                 y_bus.angle[data.pq['bus_index'][i],
                                             data.pq['bus_index'][j]] +
                                 data.pq['delta'][j] -
                                 data.pq['delta'][i])

                # PV data
                if i < data.pv.shape[0]:
                    data.pv['p_calc'][i] += data.pv['v'][i] \
                        * data.pq['v'][j] \
                        * y_bus.mag[data.pv['bus_index'][i],
                                    data.pq['bus_index'][j]] \
                        * np.cos(
                                 y_bus.angle[data.pv['bus_index'][i],
                                             data.pq['bus_index'][j]] +
                                 data.pq['delta'][j] -
                                 data.pv['delta'][i])

                    data.pv['q_calc'][i] += data.pv['v'][i] \
                        * data.pq['v'][j] \
                        * y_bus.mag[data.pv['bus_index'][i],
                                    data.pq['bus_index'][j]] \
                        * np.sin(
                                 y_bus.angle[data.pv['bus_index'][i],
                                             data.pq['bus_index'][j]] +
                                 data.pq['delta'][j] -
                                 data.pv['delta'][i])

            # PV bus
            if j < data.pv.shape[0]:
                # Slack data
                if i == 0:
                    data.slack['p_calc'] += data.slack['v'] \
                        * data.pv['v'][j] \
                        * y_bus.mag[data.slack['bus_index'],
                                    data.pv['bus_index'][j]] \
                        * np.cos(
                                 y_bus.angle[data.slack['bus_index'],
                                             data.pv['bus_index'][j]] +
                                 data.pv['delta'][j] -
                                 data.slack['delta'])

                    data.slack['q_calc'] += data.slack['v'] \
                        * data.pv['v'][j] \
                        * y_bus.mag[data.slack['bus_index'],
                                    data.pv['bus_index'][j]] \
                        * np.sin(
                                 y_bus.angle[data.slack['bus_index'],
                                             data.pv['bus_index'][j]] +
                                 data.pv['delta'][j] -
                                 data.slack['delta'])

                # PQ data
                if i < data.pq.shape[0]:
                    data.pq['p_calc'][i] += data.pq['v'][i] \
                        * data.pv['v'][j] \
                        * y_bus.mag[data.pq['bus_index'][i],
                                    data.pv['bus_index'][j]] \
                        * np.cos(
                                 y_bus.angle[data.pq['bus_index'][i],
                                             data.pv['bus_index'][j]] +
                                 data.pv['delta'][j] -
                                 data.pq['delta'][i])

                    data.pq['q_calc'][i] += data.pq['v'][i] \
                        * data.pv['v'][j] \
                        * y_bus.mag[data.pq['bus_index'][i],
                                    data.pv['bus_index'][j]] \
                        * np.sin(
                                 y_bus.angle[data.pq['bus_index'][i],
                                             data.pv['bus_index'][j]] +
                                 data.pv['delta'][j] -
                                 data.pq['delta'][i])

                # PV data
                if i < data.pv.shape[0] and i != j:
                    data.pv['p_calc'][i] += data.pv['v'][i] \
                        * data.pv['v'][j] \
                        * y_bus.mag[data.pv['bus_index'][i],
                                    data.pv['bus_index'][j]] \
                        * np.cos(
                                 y_bus.angle[data.pv['bus_index'][i],
                                             data.pv['bus_index'][j]] +
                                 data.pv['delta'][j] -
                                 data.pv['delta'][i])

                    data.pv['q_calc'][i] += data.pv['v'][i] \
                        * data.pv['v'][j] \
                        * y_bus.mag[data.pv['bus_index'][i],
                                    data.pv['bus_index'][j]] \
                        * np.sin(
                                 y_bus.angle[data.pv['bus_index'][i],
                                             data.pv['bus_index'][j]] +
                                 data.pv['delta'][j] -
                                 data.pv['delta'][i])

    p_calc = np.concatenate([
        [data.slack['p_calc']],
        data.pq['p_calc'],
        data.pv['p_calc']])
    q_calc = - np.concatenate([
        [data.slack['q_calc']],
        data.pq['q_calc'],
        data.pv['q_calc']])

    return p_calc, q_calc

def delta_p_q(data, p_calc, q_calc):
    # x_calc (every row) = [bus_index, bus_number, x]
    # x_prog (every row) = [x]' (already indexed)
    # Both vectors defined before in order (Rows order):
    # [Slack bus | PQ buses | PV buses]

    # p and q vectors
    p_prog = np.concatenate([
        [data.slack['p_prog']],
        data.pq['p_prog'],
        data.pv['p_prog']])

    q_prog = np.concatenate([
        [data.slack['q_prog']],
        data.pq['q_prog'],
        data.pv['q_prog']])

    delta_p = p_prog - p_calc
    delta_q = q_prog - q_calc

    return delta_p, delta_q

def cut_delta_p_q(data, full_delta_p, full_delta_q):
    # Cut slack data from full_delta_p
    delta_p = full_delta_p[1:]

    # Cut slack data and pv data from full_delta_q
    delta_q = full_delta_q[1:data.pq.shape[0] + 1]

    return np.concatenate((delta_p, delta_q))

def m_n_matrix(data, y_bus, p_calc, q_calc):
    # x_calc (every row) = [bus_index, bus_number, x]
    # x_calc (rows order) = [Slack_bus | PQ_buses | PV_buses]'
    # x = p or q
    p_calc_m = p_calc[1:]
    q_calc_m = q_calc[1:]
    v_m = np.concatenate([data.pq['v'], data.pv['v']])
    delta_m = np.concatenate([data.pq['delta'], data.pv['delta']])
    index_m = np.concatenate([data.pq['bus_index'], data.pv['bus_index']])

    # M matrix COO matrix specification
    row_m = []
    col_m = []
    data_m = []

    # N matrix COO matrix specification
    row_n = []
    col_n = []
    data_n = []

    # Global loop
    for i in range(p_calc_m.shape[0]):
        for j in range(p_calc_m.shape[0]):
            # mag/angle set
            if i != j and y_bus.mag[index_m[i], index_m[j]] != 0:
                # M matrix
                row_m.append(i)
                col_m.append(j)
                data_m.append(
                              - v_m[i] *
                              v_m[j] *
                              y_bus.mag[index_m[i], index_m[j]] *
                              np.sin(
                                     y_bus.angle[index_m[i], index_m[j]] +
                                     delta_m[j] -
                                     delta_m[i]))

                # N matrix
                row_n.append(i)
                col_n.append(j)
                data_n.append(
                              - v_m[i] *
                              v_m[j] *
                              y_bus.mag[index_m[i], index_m[j]] *
                              np.cos(
                                     y_bus.angle[index_m[i], index_m[j]] +
                                     delta_m[j] -
                                     delta_m[i]))

            # real/imag diag set
            elif i == j:
                # M matrix
                row_m.append(i)
                col_m.append(i)
                data_m.append(
                              - q_calc_m[i] -
                              v_m[i] ** 2 *
                              y_bus.imag[index_m[i], index_m[i]])

                # N matrix
                row_n.append(i)
                col_n.append(i)
                data_n.append(
                              p_calc_m[i] -
                              v_m[i] ** 2 *
                              y_bus.real[index_m[i], index_m[i]])

    m_mtx = scipy.sparse.coo_matrix((data_m, (row_m, col_m)),
                                    shape=(p_calc_m.shape[0],
                                    p_calc_m.shape[0])).tocsr()
    n_mtx = scipy.sparse.coo_matrix((data_n, (row_n, col_n)),
                                    shape=(p_calc_m.shape[0],
                                    p_calc_m.shape[0])).tocsr()

    return m_mtx, n_mtx

def j_construction(data, y_bus, m_mtx, n_mtx):

    x = scipy.sparse.lil_matrix((n_mtx.shape[0], n_mtx.shape[0]))
    y = copy.deepcopy(m_mtx).tolil()

    # v (every row) = [bus_index, bus_number, v] (All buses)
    # v (rows order) = [Slack_bus | PQ_buses | PV_buses]
    v_j = np.concatenate([data.pq['v'], data.pv['v']])
    index_j = np.concatenate([data.pq['bus_index'], data.pv['bus_index']])

    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            if i == j:
                x[i, j] = n_mtx[i, j] + 2 * v_j[i] ** 2 * \
                    y_bus.real[index_j[i], index_j[j]]

                y[i, j] = - m_mtx[i, j] - 2 * v_j[i] ** 2 * \
                    y_bus.imag[index_j[i], index_j[j]]
            else:
                x[i, j] = - n_mtx[i, j]

    j_a = scipy.sparse.hstack((m_mtx, x))
    j_b = scipy.sparse.hstack((n_mtx, y))

    jacobian = scipy.sparse.vstack((j_a, j_b)).tocsr()

    # Cuting PV buses to avoid delta_q calculation
    # j with all PV and PQ buses
    custom_j = jacobian[:- data.pv.shape[0], :- data.pv.shape[0]]
    j_inv = scipy.sparse.linalg.spsolve(custom_j, np.eye(custom_j.shape[0]))
    return j_inv

def sv_update(data, delta_sv):
    for i in range(max(data.pq.shape[0], data.pv.shape[0])):

        if i < data.pq.shape[0]:
            data.pq['delta'][i] = data.pq['delta'][i] + delta_sv[i]

            data.pq['v'][i] = data.pq['v'][i] * \
                (1 + delta_sv[i + data.pq.shape[0] + data.pv.shape[0]])

        if i < data.pv.shape[0]:
            data.pv['delta'][i] = data.pv['delta'][i] + \
                delta_sv[i + data.pq.shape[0]]

def sol_bus_solution(data):
    d_grad_1 = data.slack['delta'] * 180 / np.pi
    d_grad_2 = data.pq['delta'] * 180 / np.pi
    d_grad_3 = data.pv['delta'] * 180 / np.pi

    sol_1 = np.array([[
        data.slack['bus_index'],
        data.slack['bus_number'],
        data.slack['v'],
        d_grad_1,
        data.slack['base_kv'],
        data.slack['p_calc'],
        data.slack['q_calc']
    ]])

    sol_2 = np.concatenate((
        data.pq['bus_index'][np.newaxis],
        data.pq['bus_number'][np.newaxis],
        data.pq['v'][np.newaxis],
        d_grad_2[np.newaxis],
        data.pq['base_kv'][np.newaxis],
        data.pq['p_calc'][np.newaxis],
        data.pq['q_calc'][np.newaxis]
        )).T

    sol_3 = np.concatenate((
        data.pv['bus_index'][np.newaxis],
        data.pv['bus_number'][np.newaxis],
        data.pv['v'][np.newaxis],
        d_grad_3[np.newaxis],
        data.pv['base_kv'][np.newaxis],
        data.pv['p_calc'][np.newaxis],
        data.pv['q_calc'][np.newaxis]
        )).T

    bus_solution = np.concatenate((sol_1, sol_2, sol_3))
    base_kv_dtype = 'f8'

    if np.all(data.slack['base_kv'] == 0) and \
        np.all(data.pq['base_kv'] == 0) and \
            np.all(data.pq['base_kv'] == 0):

        base_kv_dtype = 'i'

    bus_solution_sort = np.array([(0, 0, 0, 0, 0, 0, 0)] * data.size,
                                 dtype=[('bus_index', 'i'),
                                 ('bus_number', 'i'),
                                 ('v', 'f8'),
                                 ('delta', 'f8'),
                                 ('base_kv', base_kv_dtype),
                                 ('p_calc', 'f8'),
                                 ('q_calc', 'f8')])

    for i in range(data.size):
        bus_index_i = np.int(bus_solution[i, 0])
        bus_solution_sort[:][bus_index_i] = tuple(bus_solution[i, :])

    return bus_solution_sort

def q_limit_consider(data, data_ini, mva_base, iteration, 
                     changed_buses, changed_voltages):

    if iteration == 0:
        for i in range(data.pv.shape[0]):
            value=False
            if data.pv['min_limit'][i] == 0 and data.pv['max_limit'][i] == 0:
                print('Warning: Unlimited Mvar generation definded for PV'
                      'bus_number:', data.pv['bus_number'][i])
                value = True

            if data.pv['min_limit'][i] >= data.pv['max_limit'][i] and \
                    value == False:
                raise LoadFlowDataError('Error: max_limit is equal or less '
                                        'than min_limit for PV bus_number: ',
                                        data.pv['bus_number'][i])

    pv_row = 0
    for i in range(data.pv.shape[0]):
        if data.pv['q_calc'][pv_row] * mva_base + \
                data.pv['q_demand'][pv_row] > data.pv['max_limit'][pv_row]:

            changed_buses.append(data.pv['bus_index'][pv_row])

            print('Warning: PV bus_number: ',
                  data.pv['bus_number'][pv_row],
                  'Mvar max_limit exceeded, changed to PQ bus')

            for row in range(data_ini.pv.shape[0]):
                if data_ini.pv['bus_index'][row] == \
                        data.pv['bus_index'][pv_row]:
                    changed_voltages.append(data_ini.pv['v'][row])

            data.pq.resize(data.pq.shape[0] + 1)
            data.pq[- 1] = data.pv[pv_row]
            data.pq['q_prog'][- 1] = (data.pq['max_limit'][- 1] -
                                      data.pq['q_demand'][- 1]) / mva_base
            data.pv = np.delete(data.pv, pv_row)
            pv_row -= 1

        if data.pv['q_calc'][pv_row] * mva_base + \
                data.pv['q_demand'][pv_row] < data.pv['min_limit'][pv_row]:

            changed_buses.append(data.pv['bus_index'][pv_row])

            print('Warning: PV bus_number: ',
                  data.pv['bus_number'][pv_row],
                  'Mvar under min_limit, changed to PQ bus')

            for row in range(data_ini.pv.shape[0]):
                if data_ini.pv['bus_index'][row] == \
                        data.pv['bus_index'][pv_row]:
                    changed_voltages.append(data_ini.pv['v'][row])

            data.pq.resize(data.pq.shape[0] + 1)
            data.pq[- 1] = data.pv[pv_row]
            data.pq['q_prog'][- 1] = (data.pq['min_limit'][- 1] -
                                      data.pq['q_demand'][- 1]) / mva_base
            data.pv = np.delete(data.pv, pv_row)
            pv_row -= 1

        pv_row += 1

    ch_row = 0
    for i in range(len(changed_buses)):
        pq_row = 0
        for j in range(data.pq.shape[0]):
            if changed_buses[ch_row] == data.pq['bus_index'][pq_row]:
                if data.pq['q_calc'][pq_row] * mva_base + \
                        data.pq['q_demand'][pq_row] >= \
                        data.pq['min_limit'][pq_row] and \
                        data.pq['q_calc'][pq_row] * mva_base + \
                        data.pq['q_demand'][pq_row] <= \
                        data.pq['max_limit'][pq_row]:
                    
                    print('Warning: PV bus_number: ',
                    data.pq['bus_number'][pq_row],
                    'Mvar between limits, returned to PV bus')
                    data.pv.resize(data.pv.shape[0] + 1)
                    data.pv[- 1] = data.pq[pq_row]
                    data.pv['v'][- 1] = changed_voltages[ch_row]
                    data.pq = np.delete(data.pq, pq_row)
                    changed_buses.remove(changed_buses[ch_row])
                    changed_voltages.remove(changed_voltages[ch_row])
                    ch_row -= 1
                    pq_row -= 1
            pq_row += 1
        ch_row += 1

    return changed_buses, changed_voltages

def v_control_data(data, target_tol, branch_data, iteration):
    # index : refered to y_bus indexation
    # position: refered to any vector position

    transf_data = []
    controled_buses = []

    # finding tap_bus_index and z_bus_index of the transformer with v-control
    branch_data_pos = 0 # Position on branch_data
    action = False  # No tap must be modified
    for i in range(len(branch_data)):
        type_i = branch_data['type'][i]
        if type_i == 2 and branch_data['min_limit'][i] > 0 and \
            branch_data['min_limit'][i] > 0:  # transformer with bus v-control plus verification
            
            side = branch_data['side'][i]  # 0: any control, 1: tap_bus controlled, 2: z_bus controlled 
            min_v = branch_data['min_limit'][i]  # min desirable voltage [p.u.]
            max_v = branch_data['max_limit'][i]  # max desirable voltage [p.u.]
            target_voltage = (min_v + max_v) / 2

            # if controled bus is near the tap_side
            if side == 1:
                control_bus_index = branch_data['tap_bus_index'][i]
                control_bus_number = branch_data['tap_bus_number'][i]
                free_bus_index = branch_data['z_bus_index'][i]
            # if controled bus is near the z_side
            elif side in [0, 2]:
                control_bus_index = branch_data['z_bus_index'][i]
                control_bus_number = branch_data['z_bus_number'][i]
                free_bus_index = branch_data['tap_bus_index'][i]
            
            if iteration == 0:
                controled_buses.append(control_bus_number)

            controled_bus = 'pv'
            free_bus = 'pv'

            for row_pq in range(data.pq.shape[0]):
                if data.pq['bus_index'][row_pq] == control_bus_index:
                    controled_v_value = data.pq['v'][row_pq]
                    pos_pq_control = row_pq  # same position in delta
                    controled_bus = 'pq'
                if data.pq['bus_index'][row_pq] == free_bus_index:
                    pos_pq_free = row_pq  # same position in delta
                    free_bus = 'pq'

            if controled_bus == 'pq':  # not a PV bus is controled
                if free_bus == 'pv':
                    for row_pv in range(data.pv.shape[0]):
                        if data.pv['bus_index'][row_pv] == free_bus_index:
                            pos_pv_free = row_pv  # same position in delta

                v_error = np.abs((controled_v_value - target_voltage) / 
                                 target_voltage)

                if (v_error >= target_tol or controled_v_value > max_v or \
                    controled_v_value < min_v):#  Tap change needed
                    
                    action = True
                    r = branch_data['resistance'][i]
                    x = branch_data['reactance'][i]
                    transf_tr = branch_data['transf_tr'][i]  # actual iteration turns_ratio
                    transf_angle = branch_data['transf_ang'][i] / 180  # actual iteration phase angle
                    transf_mintap = branch_data['transf_mintap'][i]  # min turns ratio available
                    transf_maxtap = branch_data['transf_maxtap'][i]  # max turns ratio available
                    g = r / (r ** 2 + x ** 2)  # conductance
                    b = - x / (r ** 2 + x ** 2)  # admittance
                    transf_row = (branch_data_pos, 
                                  control_bus_index, 
                                  free_bus_index, 
                                  side, 
                                  g, 
                                  b, 
                                  transf_tr, 
                                  transf_angle,
                                  transf_mintap, 
                                  transf_maxtap, 
                                  min_v, 
                                  max_v, 
                                  controled_v_value, 
                                  pos_pq_control, 
                                  pos_pq_free if free_bus == 'pq' else pos_pv_free,
                                  free_bus,
                                  0)

                    transf_data.append(transf_row)

        branch_data_pos += 1

    dtype = [('branch_data_pos', 'i'),
             ('control_bus_index', 'i'),
             ('free_bus_index', 'i'),
             ('side', 'i'),
             ('g', 'f8'),
             ('b', 'f8'),
             ('transf_tr', 'f8'),
             ('transf_angle', 'f8'),
             ('transf_mintap', 'f8'),
             ('transf_maxtap', 'f8'),
             ('min_v', 'f8'),
             ('max_v', 'f8'),
             ('controled_v_value', 'f8'),
             ('pos_pq_control', 'i'),
             ('pos_free', 'i'),
             ('free_bus_type', 'a25'),
             ('sf', 'f8')]  #sensitivity value initialy 0

    transf_data = np.array(transf_data, dtype=dtype)
    # import pdb; pdb.set_trace()
    return transf_data, action, controled_buses

def v_control_sensitivity(data, j_inv, transf_data, action):
    
    if action == True:  # At least one transformer tap must be modifiied
        sf_column = []
        tot_pq_buses = data.pq.shape[0]
        tot_pv_buses = data.pv.shape[0]
        
        for i in range(transf_data.shape[0]):
            n = 1 / transf_data['transf_tr'][i]
            n_angle = - transf_data['transf_angle'][i]
            g = transf_data['g'][i]
            b = transf_data['b'][i]
            y = np.sqrt(g ** 2 + b ** 2)
            y_angle = np.arctan2(b, g)

            dg_dui = np.zeros((2 * tot_pq_buses + tot_pv_buses, 1), 
                              dtype = 'f8')

            # Next definitions detect position on state variable vector
            control_bus_sv_pos_p = transf_data['pos_pq_control'][i]
            control_bus_sv_pos_q = transf_data['pos_pq_control'][i] + \
                                   tot_pq_buses + tot_pv_buses

            if transf_data['free_bus_type'][i] == b'pq':  # free bus is type PQ
                free_bus_sv_pos_p = transf_data['pos_free'][i]
                free_bus_sv_pos_q = transf_data['pos_free'][i] + \
                                    tot_pq_buses + tot_pv_buses
            else:  # free bus is type PV
                free_bus_sv_pos_p = transf_data['pos_free'][i] + \
                                    tot_pq_buses
            
            pos_free = transf_data['pos_free'][i]
            if transf_data['free_bus_type'][i] == b'pq':
                v_i = data.pq['v'][pos_free]
                delta_i = data.pq['delta'][pos_free]
            else:
                v_i = data.pv['v'][pos_free]
                delta_i = data.pv['delta'][pos_free]

            pos_control = transf_data['pos_pq_control'][i]
            v_j = data.pq['v'][pos_control]
            delta_j = data.pq['delta'][pos_control]
        
            dg_dui[free_bus_sv_pos_p, 0] = - 2 * n * v_j ** 2 * g \
                                           + v_i * v_j * y *  np.cos(delta_i 
                                                                 - delta_j
                                                                 - y_angle 
                                                                 + n_angle)

            dg_dui[control_bus_sv_pos_p, 0] = v_i * v_j * y *  np.cos(delta_j 
                                                                  - delta_i
                                                                  - y_angle 
                                                                  + n_angle)
            
            if transf_data['free_bus_type'][i] == 'pq':
                dg_dui[free_bus_sv_pos_q, 0] = 2 * n * v_i ** 2 * b \
                                               + v_i * v_j * y *  np.sin(delta_i 
                                                                     - delta_j
                                                                     - y_angle 
                                                                     + n_angle)

            dg_dui[control_bus_sv_pos_q, 0] = v_i * v_j * y * np.sin(delta_j 
                                                                  - delta_i
                                                                  - y_angle 
                                                                  + n_angle)

            j_inv_cut = j_inv[control_bus_sv_pos_q, :]
            sf_value = j_inv_cut.dot(dg_dui)  # sensitivity value sf = 
            transf_data['sf'][i] = sf_value

    return transf_data

def y_bus_mod_tap_transf(y_bus, tap_bus_index, z_bus_index, g,
                         b, t, t_angle, action):

    t_real = t * np.cos(t_angle)
    t_imag = t * np.sin(t_angle)

    g_tap_tap = g / (t ** 2)
    g_z_z = g
    g_tap_z = - (g * t_real - b * t_imag) / (t ** 2)
    g_z_tap = - (g * t_real + b * t_imag) / (t ** 2)

    b_tap_tap = b / (t ** 2)
    b_z_z = b
    b_tap_z = - (g * t_imag + b * t_real) / (t ** 2)
    b_z_tap = - (- g * t_imag + b * t_real) / (t ** 2)
    
    # Deleting transformer with old tap

    # Diagonal elements
    y_bus.real[tap_bus_index, tap_bus_index] += action * g_tap_tap
    y_bus.imag[tap_bus_index, tap_bus_index] += action * b_tap_tap

    y_bus.real[z_bus_index, z_bus_index] += action * g_z_z
    y_bus.imag[z_bus_index, z_bus_index] += action * b_z_z

    # Non diagonal elements, asymmetric y_bus !
    y_bus.real[tap_bus_index, z_bus_index] += action * g_tap_z
    y_bus.imag[tap_bus_index, z_bus_index] += action * b_tap_z

    y_bus.real[z_bus_index, tap_bus_index] += action * g_z_tap
    y_bus.imag[z_bus_index, tap_bus_index] += action * b_z_tap
    
    if action == 1:
        y_bus.mag[tap_bus_index, tap_bus_index] = np.sqrt(y_bus.real[tap_bus_index, tap_bus_index] ** 2
                                                       + y_bus.imag[tap_bus_index, tap_bus_index] ** 2)
        y_bus.angle[tap_bus_index, tap_bus_index] = np.arctan2(y_bus.imag[tap_bus_index, tap_bus_index],
                                                          y_bus.real[tap_bus_index, tap_bus_index])

        y_bus.mag[z_bus_index, z_bus_index] = np.sqrt(y_bus.real[z_bus_index, z_bus_index] ** 2
                                                       + y_bus.imag[z_bus_index, z_bus_index] ** 2)
        y_bus.angle[z_bus_index, z_bus_index] = np.arctan2(y_bus.imag[z_bus_index, z_bus_index],
                                                          y_bus.real[z_bus_index, z_bus_index])

        y_bus.mag[tap_bus_index, z_bus_index] = np.sqrt(y_bus.real[tap_bus_index, z_bus_index] ** 2
                                                       + y_bus.imag[tap_bus_index, z_bus_index] ** 2)
        y_bus.angle[tap_bus_index, z_bus_index] = np.arctan2(y_bus.imag[tap_bus_index, z_bus_index],
                                                          y_bus.real[tap_bus_index, z_bus_index])

        y_bus.mag[z_bus_index, tap_bus_index] = np.sqrt(y_bus.real[z_bus_index, tap_bus_index] ** 2
                                                       + y_bus.imag[z_bus_index, tap_bus_index] ** 2)
        y_bus.angle[z_bus_index, tap_bus_index] = np.arctan2(y_bus.imag[z_bus_index, tap_bus_index],
                                                          y_bus.real[z_bus_index, tap_bus_index])

def v_control_tap_update(data, y_bus, branch_data, transf_data, tol_tr):

    for i in range(transf_data.shape[0]):
        # sf_data (columns) = [position, control_bus_index, free_bus_index, side, g, b, transf_tr, transf_ang, \
        #                        transf_mintap, transf_maxtap, min_v, max_v, controled_voltage_value, sf_value]
        position = transf_data['branch_data_pos'][i]
        control_bus_index = transf_data['control_bus_index'][i]
        free_bus_index = transf_data['free_bus_index'][i]
        side = transf_data['side'][i]
        g = transf_data['g'][i]
        b = transf_data['b'][i]
        t = transf_data['transf_tr'][i]
        t_angle = transf_data['transf_angle'][i]
        transf_mintap = transf_data['transf_mintap'][i]
        transf_maxtap = transf_data['transf_maxtap'][i]
        min_v = transf_data['min_v'][i]
        max_v = transf_data['max_v'][i]
        voltage = transf_data['controled_v_value'][i]
        sf_value = transf_data['sf'][i]

        tr_range = transf_maxtap - transf_mintap
        target_voltage = (min_v + max_v) / 2

        if side == 1:
            tap_bus_index = control_bus_index
            z_bus_index = free_bus_index

        elif side in [0, 2]:
            tap_bus_index = free_bus_index
            z_bus_index = control_bus_index

        if abs(sf_value) > 0:
            delta_n = (target_voltage - voltage) / sf_value
        else:
            delta_n = 0

        new_n = 1 / t + delta_n
        new_t = 1 / new_n
        
        required_change = abs((new_t - t) / tr_range)

        if required_change > tol_tr:
            if new_t - t > 0:
                new_t = t + tol_tr * tr_range

            elif new_t - t < 0:
                new_t = t - tol_tr * tr_range

        if new_t > transf_maxtap:
            new_t = transf_maxtap
        if new_t < transf_mintap:
            new_t = transf_mintap
        
        if t != new_t:
            # Removing transformer with actual tap
            y_bus_mod_tap_transf(y_bus, tap_bus_index, z_bus_index, g, b, t,
                                 t_angle, - 1)
            
            # Adding transformer with new
            y_bus_mod_tap_transf(y_bus, tap_bus_index, z_bus_index, g, b, new_t, 
                                 t_angle, + 1)

            branch_data['transf_tr'][position] = new_t

def sol_line_flows(bus_solution, bus_data, branch_data, mva_base):

    pre_line_flows = []

    for i in range(branch_data.__len__()):
        if branch_data['type'][i] == 0:  # transmission line

            g_series = branch_data['resistance'][i] / \
                (branch_data['resistance'][i] ** 2 +
                 branch_data['reactance'][i] ** 2)

            b_series = - branch_data['reactance'][i] / \
                (branch_data['resistance'][i] ** 2 +
                 branch_data['reactance'][i] ** 2)

            b_shunt = branch_data['line_charging'][i]

            from_bus = branch_data['tap_bus_index'][i]
            to_bus = branch_data['z_bus_index'][i]

            v_from_real = bus_solution['v'][from_bus] * \
                np.cos(bus_solution['delta'][from_bus] * np.pi / 180)

            v_from_imag = bus_solution['v'][from_bus] * \
                np.sin(bus_solution['delta'][from_bus] * np.pi / 180)

            v_to_real = bus_solution['v'][to_bus] * \
                np.cos(bus_solution['delta'][to_bus] * np.pi / 180)

            v_to_imag = bus_solution['v'][to_bus] * \
                np.sin(bus_solution['delta'][to_bus] * np.pi / 180)

            delta_v_real = v_from_real - v_to_real
            delta_v_imag = v_from_imag - v_to_imag

            i_from_to_real = delta_v_real * g_series - delta_v_imag * b_series
            i_from_to_imag = delta_v_real * b_series + delta_v_imag * g_series

            i_from_shunt_real = - v_from_imag * b_shunt / 2
            i_from_shunt_imag = v_from_real * b_shunt / 2

            i_to_shunt_real = - v_to_imag * b_shunt / 2
            i_to_shunt_imag = v_to_real * b_shunt / 2

            i_from_real = i_from_to_real + i_from_shunt_real
            i_from_imag = i_from_to_imag + i_from_shunt_imag
            i_from_mag = np.sqrt(i_from_real ** 2 + i_from_imag ** 2)
            i_from_angle = np.arctan2(i_from_imag, i_from_real) * 180 / np.pi

            i_to_real = i_from_to_real - i_to_shunt_real
            i_to_imag = i_from_to_imag - i_to_shunt_imag
            i_to_mag = np.sqrt(i_to_real ** 2 + i_to_imag ** 2)
            i_to_angle = np.arctan2(i_to_imag, i_to_real) * 180 / np.pi

            # Power flow through lines
            p_from = v_from_real * i_from_real + v_from_imag * i_from_imag
            q_from = - v_from_real * i_from_imag + v_from_imag * i_from_real

            p_to = v_to_real * i_to_real + v_to_imag * i_to_imag
            q_to = - v_to_real * i_to_imag + v_to_imag * i_to_real

            trigger = 0
            if bus_data['base_kv'][from_bus] == 0 or \
                    bus_data['base_kv'][to_bus] == 0:
                trigger = 1
                base_ka = 0

            if trigger == 0:
                base_ka_1 = mva_base / (np.sqrt(3) *
                                        bus_data['base_kv'][from_bus])

                base_ka_2 = mva_base / (np.sqrt(3) *
                                        bus_data['base_kv'][to_bus])

                if base_ka_1 != base_ka_2:
                    raise LoadFlowDataError('ERROR: Base voltages defined for'
                                            'transmission line between buses'
                                            '[',
                                            branch_data['tap_bus_number'][i],
                                            ', ',
                                            branch_data['z_bus_number'][i],
                                            '] are not equal.')

                base_ka = base_ka_1

            row = (branch_data['tap_bus_number'][i],
                   branch_data['z_bus_number'][i],
                   branch_data['area_number'][i],  # In case of using line_id
                   i_from_mag,
                   i_from_angle,
                   p_from,
                   q_from,
                   i_to_mag,
                   i_to_angle,
                   p_to,
                   q_to,
                   base_ka)

            pre_line_flows.append(row)

    base_ka_dtype = 'f8'
    if trigger == 1:
        base_ka_dtype = 'i'

    line_flows = np.array(pre_line_flows, dtype=[('tap_bus_number', 'i'),
                                                 ('z_bus_number', 'i'),
                                                 ('area_number', 'i'),
                                                 ('current_from_mag', 'f8'),
                                                 ('current_from_angle', 'f8'),
                                                 ('p_from', 'f8'),
                                                 ('q_from', 'f8'),
                                                 ('current_to_mag', 'f8'),
                                                 ('current_to_angle', 'f8'),
                                                 ('p_to', 'f8'),
                                                 ('q_to', 'f8'),
                                                 ('base_ka', base_ka_dtype)])

    return line_flows

def sol_transformer_flows(bus_solution, bus_data, branch_data, mva_base):

    pre_transformer_flows = []

    for i in range(branch_data.__len__()):
        if branch_data['type'][i] in [1, 2, 3, 4]:  # transformer

            g_series = branch_data['resistance'][i] / \
                (branch_data['resistance'][i] ** 2 +
                 branch_data['reactance'][i] ** 2)

            b_series = - branch_data['reactance'][i] / \
                (branch_data['resistance'][i] ** 2 +
                 branch_data['reactance'][i] ** 2)

            t_mag = branch_data['transf_tr'][i]
            if t_mag == 0:
                t_mag = 1
            t_angle = branch_data['transf_ang'][i]
            t_real = t_mag * np.cos(t_angle * np.pi / 180)
            t_imag = t_mag * np.sin(t_angle * np.pi / 180)

            from_bus = branch_data['tap_bus_index'][i]
            to_bus = branch_data['z_bus_index'][i]

            v_from_real = bus_solution['v'][from_bus] * \
                np.cos(bus_solution['delta'][from_bus] *
                       np.pi / 180)

            v_from_imag = bus_solution['v'][from_bus] * \
                np.sin(bus_solution['delta'][from_bus] *
                       np.pi / 180)

            ref_v_from_real = (v_from_real * t_real +
                               v_from_imag * t_imag) / t_mag ** 2
            ref_v_from_imag = (v_from_imag * t_real -
                               v_from_real * t_imag) / t_mag ** 2

            v_to_real = bus_solution['v'][to_bus] * \
                np.cos(bus_solution['delta'][to_bus] *
                       np.pi / 180)

            v_to_imag = bus_solution['v'][to_bus] * \
                np.sin(bus_solution['delta'][to_bus] *
                       np.pi / 180)

            delta_v_real = ref_v_from_real - v_to_real
            delta_v_imag = ref_v_from_imag - v_to_imag

            ref_i_from_to_real = delta_v_real * g_series - delta_v_imag * \
                b_series

            ref_i_from_to_imag = delta_v_real * b_series + delta_v_imag * \
                g_series

            ref_i_from_to_mag = np.sqrt(ref_i_from_to_real ** 2 +
                                        ref_i_from_to_imag ** 2)
            ref_i_from_to_angle = np.arctan2(ref_i_from_to_imag,
                                             ref_i_from_to_real) * np.pi / 180

            i_from_to_real = (ref_i_from_to_real * t_real -
                              ref_i_from_to_imag * t_imag) / t_mag ** 2
            i_from_to_imag = (ref_i_from_to_real * t_imag +
                              ref_i_from_to_imag * t_real) / t_mag ** 2

            i_from_to_mag = np.sqrt(i_from_to_real ** 2 + i_from_to_imag ** 2)

            i_from_to_angle = np.arctan2(i_from_to_imag, i_from_to_real) \
                * 180 / np.pi

            ref_i_from_to_mag = np.sqrt(ref_i_from_to_real ** 2 +
                                        ref_i_from_to_imag ** 2)

            ref_i_from_to_angle = np.arctan2(ref_i_from_to_imag,
                                             ref_i_from_to_real) * 180 / np.pi

            p_from = v_from_real * i_from_to_real + v_from_imag * \
                i_from_to_imag

            q_from = v_from_imag * i_from_to_real - v_from_real * \
                i_from_to_imag

            p_to = v_to_real * ref_i_from_to_real + v_to_imag * \
                ref_i_from_to_imag
            q_to = v_to_imag * ref_i_from_to_real - v_to_real * \
                ref_i_from_to_imag

            trigger = 0
            if bus_data['base_kv'][from_bus] == 0 or \
                    bus_data['base_kv'][to_bus] == 0:
                trigger = 1
                base_ka_tap = 0
                base_ka_z = 0

            if trigger == 0:
                base_ka_tap = mva_base / (np.sqrt(3) *
                                          bus_data['base_kv'][from_bus])
                base_ka_z = mva_base / (np.sqrt(3) *
                                        bus_data['base_kv'][to_bus])

            row = (branch_data['tap_bus_number'][i],
                   branch_data['z_bus_number'][i],
                   branch_data['area_number'][i],  # In case of using line_id
                   branch_data['type'][i],
                   i_from_to_mag,
                   i_from_to_angle,
                   base_ka_tap,
                   p_from,
                   q_from,
                   ref_i_from_to_mag,
                   ref_i_from_to_angle,
                   base_ka_z,
                   p_to,
                   q_to)

            pre_transformer_flows.append(row)

    base_ka_tap_dtype = 'f8'
    base_ka_z_dtype = 'f8'
    if trigger == 1:
        base_ka_tap_dtype = 'i'
        base_ka_z_dtype = 'i'

    transformer_flows = np.array(pre_transformer_flows,
                                 dtype=[('tap_bus_number', 'i'),
                                        ('z_bus_number', 'i'),
                                        ('area_number', 'i'),
                                        ('type', 'i'),
                                        ('i_from_to_mag', 'f8'),
                                        ('i_from_to_angle', 'f8'),
                                        ('base_ka_tap', base_ka_tap_dtype),
                                        ('p_from', 'f8'),
                                        ('q_from', 'f8'),
                                        ('ref_i_from_to_mag', 'f8'),
                                        ('ref_i_from_to_angle', 'f8'),
                                        ('base_ka_z', base_ka_z_dtype),
                                        ('p_to', 'f8'),
                                        ('q_to', 'f8')])

    return transformer_flows

def sol_shunt_flows(bus_solution, bus_data, mva_base):

    pre_shunt_flows = []

    for bus_index in range(bus_data.__len__()):
        g = bus_data['shunt_g'][bus_index]
        b = bus_data['shunt_b'][bus_index]

        if g != 0 or b != 0:
            v_mag = bus_solution['v'][bus_index]
            v_angle = bus_solution['delta'][bus_index]
            v_real = v_mag * np.cos(v_angle * np.pi / 180)
            v_imag = v_mag * np.sin(v_angle * np.pi / 180)

            i_real = v_real * g - v_imag * b
            i_imag = v_imag * g + v_real * b
            i_mag = np.sqrt(i_real ** 2 + i_imag ** 2)
            i_angle = np.arctan2(i_imag, i_real) * 180 / np.pi

            p = v_real * i_real + v_imag * i_imag
            q = v_imag * i_real - v_real * i_imag

            trigger = 0
            if bus_data['base_kv'][bus_index] == 0:
                trigger = 1
                base_ka = 0

            if trigger == 0:
                base_ka = mva_base / (np.sqrt(3) *
                                      bus_data['base_kv'][bus_index])

            row = (bus_data['bus_number'][bus_index],
                   bus_data['area_number'][bus_index],
                   i_mag,
                   i_angle,
                   base_ka,
                   p,
                   q)

            pre_shunt_flows.append(row)

    base_ka_dtype = 'f8'
    if trigger == 1:
        base_ka_dtype = 'i'

    shunt_flows = np.array(pre_shunt_flows, dtype=[('bus_number', 'i'),
                                                   ('area_number', 'i'),
                                                   ('i_mag', 'f8'),
                                                   ('i_angle', 'f8'),
                                                   ('base_ka', base_ka_dtype),
                                                   ('p', 'f8'),
                                                   ('q', 'f8')])

    return shunt_flows

def round_solution(solution, decimals):

    """
    Round each element of the structured array solution to the desired number
    of decimals
    """

    for i in range(solution.shape[0]):
        for j in range(len(solution[i])):
            solution[i][j] = round(solution[i][j], decimals)

def write_solution(solution, sol_path):

    """
    Write a structured array solution to a csv file
    """

    with open(sol_path, 'w') as output_sol:
        csv_sol = csv.writer(output_sol)
        for row in solution:
            csv_sol.writerow(row)

def write_general_data(mva_base, time, iteration, error, general_data_path):

    """
    Write power flow general_data to a csv file
    """

    with open(general_data_path, 'w') as output_general_data:
        csv_general_data = csv.writer(output_general_data)
        csv_general_data.writerows([['MVA base', mva_base],
                                   ['Convergence Time', time],
                                   ['Iterations', iteration],
                                   ['Error', error]])


class LoadFlowData(object):
    def __init__(self, bus_data, mva_base):
        self._slack_size = 0
        self._pq_size = 0
        self._pv_size = 0

        list_pv = []
        list_pq = []

        dtype = [('bus_index', 'i'),
                 ('bus_number', 'i'),
                 ('type', 'i'),
                 ('v', 'f8'),
                 ('delta', 'f8'),
                 ('p_prog', 'f8'),
                 ('q_prog', 'f8'),
                 ('p_calc', 'f8'),
                 ('q_calc', 'f8'),
                 ('base_kv', 'f8'),
                 ('max_limit', 'f8'),
                 ('min_limit', 'f8'),
                 ('q_demand', 'f8')]

        for i in range(len(bus_data)):
            row = (i,
                   bus_data['bus_number'][i],
                   bus_data['type'][i],
                   bus_data['v'][i],
                   bus_data['delta'][i],
                   (bus_data['pg'][i] - bus_data['pd'][i]) / mva_base,
                   (bus_data['qg'][i] - bus_data['qd'][i]) / mva_base,
                   0,
                   0,
                   bus_data['base_kv'][i],
                   bus_data['max_limit'][i],
                   bus_data['min_limit'][i],
                   bus_data['qd'][i])

            # Slack_bus
            if bus_data['type'][i] == 3:
                list_slack = row
                self._slack_size += 1

            # PQ Bus
            elif bus_data['type'][i] == 0 or bus_data['type'][i] == 1:
                list_pq.append(row)
                self._pq_size += 1

            # PV Bus
            elif bus_data['type'][i] == 2:
                list_pv.append(row)
                self._pv_size += 1

        self.slack = np.array(list_slack, dtype=dtype)
        self.pq = np.array(list_pq, dtype=dtype)
        self.pv = np.array(list_pv, dtype=dtype)
        self._size = i + 1

    @property
    def size(self):
        return self._size

    @property
    def slack_size(self):
        return self._slack_size

    @property
    def pq_size(self):
        return self._pq_size

    @property
    def pv_size(self):
        return self._pv_size
