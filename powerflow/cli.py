from cement.core.foundation import CementApp
from cement.ext.ext_argparse import ArgparseController, expose
from powerflow import PowerFlow
from powerflow.exceptions import CLIError, DataException
import numpy as np


class BaseController(ArgparseController):

    class Meta:

        """
        BaseController arguments definition
        """

        label = 'base'
        description = 'To run a power flow'

    def default(self):

        """
        Print help text if no commands are provided to CLI
        """

        self.app.args.print_help()

    @expose(
        arguments=[
                   (['bus_data_path'],
                    dict(action='store',
                    help='Path to bus data file')
                    ),
                   (['branch_data_path'],
                    dict(action='store',
                    help='Path to branch data file')
                    ),
                   (['instance_name'],
                    dict(action='store',
                    help='Name of folder to put generated files')
                    ),
                   (['--force'],
                    dict(action='store',
                    help='Generate files even if they exist',
                    default='False',
                    choices=['True', 'true', 'false', 'False'])
                    )
                   ],

        help='Generates y_bus_real, y_bus_imag, y_bus_mag and y_bus_angle'
    )
    def ybc(self):

        """
        Generates y_bus_real, y_bus_imag, y_bus_mag and y_bus_angle
        3 arguments expected: bus_data_path, branch_data_path and instance name
        """

        bus_data_path = self.app.pargs.bus_data_path
        branch_data_path = self.app.pargs.branch_data_path
        instance_name = self.app.pargs.instance_name
        force = True if self.app.pargs.force == 'True' or \
            self.app.pargs.force == 'true' else False

        try:
            # If both paths are valid, construct fast y bus files
            self.app.log.info('Building y bus matrices...')
            PowerFlow(instance_name, bus_data_path, branch_data_path, force)
            self.app.log.info('Process completed for instance "' +
                              instance_name + '"')

        # Throws error if bus_data_path and/or branch_data_path aren't valid
        except BaseException as e:
            self.app.log.error(e)

    @expose(
        arguments=[
                   (['--max_iter'],
                    dict(action='store',
                         help='Maximum number of iterations to perform on'
                         'iterative process. Default: max_iter=10',
                    default=10)
                    ),
                   (['--tol'],
                    dict(action='store',
                         help='Tolerance of iterative process in Mvar or MW.'
                         'Default: tol=0.01',
                         default=0.01)
                    ),
                   (['--decimals'],
                    dict(action='store',
                         help='Number of decimals to store and print'
                         'bus_solution and line_flows. Default: decimals=4',
                    default=4)
                    ),
                   (['--dir_path'],
                    dict(action='store',
                         help='Path to required data for power'
                         'flow execution.'
                         'Default: dir_path=None',
                    default=None)
                    ),
                   (['--bus_solution'],
                    dict(action='store_true',
                         help='To only print bus_solution')
                    ),
                   (['--line_flows'],
                    dict(action='store_true',
                         help='To only print line_flows')
                    ),
                   (['--transf_flows'],
                    dict(action='store_true',
                         help='To only print transf_flows'),
                    ),
                   (['--shunt_flows'],
                    dict(action='store_true',
                         help='To only print shunt_flows'),
                    ),
                   (['--full_sol'],
                    dict(action='store_true',
                    help='To print bus_solution, line_flows,'
                    'convergence_time, number of iterations and error')
                    ),
                   (['--reactive_limits'],
                    dict(action='store',
                         help='Consider maximum and minimum reactive limits'
                         'in generator units',
                         default='False',
                         choices=['True', 'true', 'false', 'False'])
                    ),

                   (['--tap_v_control'],
                    dict(action='store',
                         help='Consider transformer tap adjust for voltage \
                               control',
                         default='False',
                         choices=['True', 'true', 'false', 'False'])
                    ),

                   (['--tap_q_control'],
                    dict(action='store',
                         help='Consider transformer tap adjust for reactive \
                               flow control', 
                         default='False',
                         choices=['True', 'true', 'false', 'False'])
                    ),

                   (['instance_name'],
                    dict(action='store',
                         help='Name of folder where y bus files are stored')
                    ),
                  ],

        help='Executes power flow for a given instance and displays solution'
    )
    def lf(self):

        """
        Run power flow for a given instance and display solution
        """

        instance_name = self.app.pargs.instance_name

        try:
            max_iter = int(self.app.pargs.max_iter)
            assert max_iter > 0
        except BaseException:
            raise CLIError('max_iter must be a positive integer number')

        try:
            tol = float(self.app.pargs.tol)
            assert tol > 0
        except BaseException:
            raise CLIError('tol must be a positive floating point value')

        try:
            self.decimals = int(self.app.pargs.decimals)
            assert self.decimals > 0
        except BaseException:
            raise CLIError('decimals must be a positive integer number')

        reactive_limits = True if self.app.pargs.reactive_limits == 'True' or \
            self.app.pargs.reactive_limits == 'true' else False

        tap_v_control = True if self.app.pargs.tap_v_control == 'True' or \
            self.app.pargs.tap_v_control == 'true' else False

        tap_q_control = True if self.app.pargs.tap_q_control == 'True' or \
            self.app.pargs.tap_q_control == 'true' else False

        dir_path = self.app.pargs.dir_path

        lf_instance = PowerFlow(instance_name)

        try:
            self.app.log.info('Running power flow...')
            lf_instance.load_flow(max_iter, tol, self.decimals, 
                                  reactive_limits, tap_v_control, 
                                  tap_q_control, dir_path)
            self.app.log.info('Power flow completed')

            self.bus_solution, time, iteration, error, self.line_flows, \
                self.transf_flows, \
                self.shunt_flows, \
                mva_base = lf_instance.get_load_flow_solution()

            if self.app.pargs.bus_solution or self.app.pargs.full_sol:
                self.print_bus_solution()

            if self.app.pargs.line_flows or self.app.pargs.full_sol:
                self.print_line_flows()

            if self.app.pargs.transf_flows or self.app.pargs.full_sol:
                self.print_transf_flows()

            if self.app.pargs.shunt_flows or self.app.pargs.full_sol:
                self.print_shunt_flows()

            if self.app.pargs.full_sol:
                print('\nmva_base[MVA]: ', mva_base, '\nConvergence time: ',
                      time, '\nNumber of iterations: ', iteration, '\nError: ',
                      error, '\n')

        except DataException as e:
            self.app.log.error(e)

    def print_bus_solution(self):
        print('\n---Bus Solution---')
        print('\tcol1: bus_number'
              '\n\tcol2: v[p.u.]'
              '\n\tcol3: delta[deg]'
              '\n\tcol4: base_voltage[kv]'
              '\n\tcol5: p_calc[p.u.]'
              '\n\tcol6: q_calc[p.u.]\n')

        for i in range(self.bus_solution.shape[0]):

            print(self.bus_solution['bus_number'][i], '\t \t',

                  '{0:0.{dec}f}'
                  .format(self.bus_solution['v'][i], dec=self.decimals),

                  '\t', '{0:0.{dec}f}'
                  .format(self.bus_solution['delta'][i], dec=self.decimals),

                  '\t', '{0:0.{dec}f}'
                  .format(self.bus_solution['base_kv'][i], dec=self.decimals)
                  if type(self.bus_solution['base_kv'][i]) == np.float64
                  else 0,

                  '\t', '{0:0.{dec}f}'
                  .format(self.bus_solution['p_calc'][i], dec=self.decimals),

                  '\t', '{0:0.{dec}f}'
                  .format(self.bus_solution['q_calc'][i], dec=self.decimals))

    def print_line_flows(self):
        print('\n---Line Flows: Currents (current direction: From -> To)---')
        print('\tcol1: From bus_number'
              '\n\tcol2: To bus_number'
              '\n\tcol3: area_number'
              '\n\tcol4: Current magnitude [p.u.], From bus side'
              '\n\tcol5: Current angle[º], From bus side'
              '\n\tcol6: Current magnitude [p.u.], To bus side'
              '\n\tcol7: Current angle[º], To bus side'
              '\n\tcol8: Base current[kA]\n')

        for i in range(self.line_flows.shape[0]):

            print(self.line_flows['tap_bus_number'][i],
                  '\t', self.line_flows['z_bus_number'][i],
                  '\t', self.line_flows['area_number'][i],

                  '\t', '{0:0.{dec}f}'
                  .format(self.line_flows['current_from_mag'][i],
                          dec=self.decimals),

                  '\t', '{0:0.{dec}f}'
                  .format(self.line_flows['current_from_angle'][i],
                          dec=self.decimals),

                  '\t', '{0:0.{dec}f}'
                  .format(self.line_flows['current_to_mag'][i],
                          dec=self.decimals),

                  '\t', '{0:0.{dec}f}'
                  .format(self.line_flows['current_to_angle'][i],
                          dec=self.decimals),

                  '\t', '{0:0.{dec}f}'
                  .format(self.line_flows['base_ka'][i], dec=self.decimals)
                  if type(self.line_flows['base_ka'][i]) == np.float64
                  else 0)

        print('\n---Line Flows: Power flows (flow direction: From -> To)---')
        print('\tcol1: From bus_number'
              '\n\tcol2: To bus_number'
              '\n\tcol3: area_number'
              '\n\tcol4: Active power flow (p) [p.u.], From bus side'
              '\n\tcol5: Reactive inductive power (q) [p.u.], From bus side'
              '\n\tcol6: Active power flow (p) [p.u.], To bus side'
              '\n\tcol7: Reactive inductive power flow (q) [p.u.],'
              ' To bus side\n')

        for i in range(self.line_flows.shape[0]):
            print(self.line_flows['tap_bus_number'][i],
                  '\t', self.line_flows['z_bus_number'][i],
                  '\t', self.line_flows['area_number'][i],

                  '\t', '{0:0.{dec}f}'
                  .format(self.line_flows['p_from'][i], dec=self.decimals),


                  '\t', '{0:0.{dec}f}'
                  .format(self.line_flows['q_from'][i], dec=self.decimals),

                  '\t', '{0:0.{dec}f}'
                  .format(self.line_flows['p_to'][i], dec=self.decimals),

                  '\t', '{0:0.{dec}f}'.format(self.line_flows['q_to'][i],
                                              dec=self.decimals))

    def print_transf_flows(self):
        print('\n---Transformer Flows: Currents (current direction: From -> To'
              ')---')
        print('\tcol1: From bus_number (tap_bus_number)'
              '\n\tcol2: To bus_number (z_bus_number)'
              '\n\tcol3: area_number'
              '\n\tcol4: type'
              '\n\tcol5: Current magnitude [p.u.], From bus side'
              '\n\tcol6: Current angle [º], From bus side'
              '\n\tcol7: Base current [kA], From bus side'
              '\n\tcol8: Current magnitude [p.u.], To bus side'
              '\n\tcol9: Current angle [º], To bus side'
              '\n\tcol10: Base current [kA], To bus side\n')

        for i in range(self.transf_flows.shape[0]):

            print(self.transf_flows['tap_bus_number'][i],
                  '\t', self.transf_flows['z_bus_number'][i],
                  '\t', self.transf_flows['area_number'][i],
                  '\t', self.transf_flows['type'][i],

                  '\t', '{0:0.{dec}f}'
                  .format(self.transf_flows['i_from_to_mag'][i],
                          dec=self.decimals),

                  '\t', '{0:0.{dec}f}'
                  .format(self.transf_flows['i_from_to_angle'][i],
                          dec=self.decimals),

                  '\t', '{0:0.{dec}f}'
                  .format(self.transf_flows['base_ka_tap'][i],
                          dec=self.decimals
                          if type(self.transf_flows['base_ka_tap'][i]) ==
                          np.float64 else 0),

                  '\t', '{0:0.{dec}f}'
                  .format(self.transf_flows['ref_i_from_to_mag'][i],
                          dec=self.decimals),

                  '\t', '{0:0.{dec}f}'
                  .format(self.transf_flows['ref_i_from_to_angle'][i],
                          dec=self.decimals),

                  '\t', '{0:0.{dec}f}'
                  .format(self.transf_flows['base_ka_z'][i], dec=self.decimals
                          if type(self.transf_flows['base_ka_z'][i]) ==
                          np.float64 else 0))

        print('\n---Transformer Flows: Power flows (flow direction: From -> To'
              ')---')
        print('\tcol1: From bus_number'
              '\n\tcol2: To bus_number'
              '\n\tcol3: area_number'
              '\n\tcol4: type'
              '\n\tcol5: Active power flow (p) [p.u.], From bus side'
              ' (tap_bus_number)'
              '\n\tcol6: Reactive inductive power (q) [p.u.], From bus side'
              ' (tap_bus_number)'
              '\n\tcol7: Active power flow (p) [p.u.], To bus side'
              ' (z_bus_number)'
              '\n\tcol8: Reactive inductive power flow (q) [p.u.], To bus side'
              ' (z_bus_number)\n')

        for i in range(self.transf_flows.shape[0]):
            print(self.transf_flows['tap_bus_number'][i],
                  '\t', self.transf_flows['z_bus_number'][i],
                  '\t', self.transf_flows['area_number'][i],
                  '\t', self.transf_flows['type'][i],

                  '\t', '{0:0.{dec}f}'
                  .format(self.transf_flows['p_from'][i], dec=self.decimals),

                  '\t', '{0:0.{dec}f}'
                  .format(self.transf_flows['q_from'][i], dec=self.decimals),

                  '\t', '{0:0.{dec}f}'
                  .format(self.transf_flows['p_to'][i], dec=self.decimals),

                  '\t', '{0:0.{dec}f}'
                  .format(self.transf_flows['q_to'][i], dec=self.decimals))

    def print_shunt_flows(self):
        print('\n---Shunt Flows (flow direction: Bus -> Earth)---')
        print('\tcol1: From bus_number'
              '\n\tcol2: area_number'
              '\n\tcol3: Current magnitude [p.u.]'
              '\n\tcol4: Current angle [º]'
              '\n\tcol5: Base current [kA]'
              '\n\tcol6: Active power flow (p) [p.u.]'
              '\n\tcol7: Reactive inductive power flow (q) [p.u.]\n')

        for i in range(self.shunt_flows.shape[0]):
            print(self.shunt_flows['bus_number'][i],
                  '\t', self.shunt_flows['area_number'][i],

                  '\t', '{0:0.{dec}f}'
                  .format(self.shunt_flows['i_mag'][i], dec=self.decimals),

                  '\t', '{0:0.{dec}f}'
                  .format(self.shunt_flows['i_angle'][i], dec=self.decimals),

                  '\t', '{0:0.{dec}f}'
                  .format(self.shunt_flows['base_ka'][i], dec=self.decimals)
                  if type(self.shunt_flows['base_ka'][i]) ==
                  np.float64 else 0,

                  '\t', '{0:0.{dec}f}'.format(self.shunt_flows['p'][i],
                                              dec=self.decimals),
                  '\t', '{0:0.{dec}f}'.format(self.shunt_flows['q'][i],
                                              dec=self.decimals))


class PowerFlowCli(CementApp):

    """
    CLI specification
    """

    class Meta:
        label = 'Powerflow-CLI'
        base_controller = 'base'
        handlers = [BaseController]


# Run CLI
def run_cli():
    with PowerFlowCli() as cli:
        cli.run()
