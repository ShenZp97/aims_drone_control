""" Implementation of the nonlinear optimizer for the data-augmented MPC.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""


import os
import sys
import shutil
import casadi as cs
import numpy as np
import errno
# import yaml
from copy import copy
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel

def skew_symmetric(v):
    """
    Computes the skew-symmetric matrix of a 3D vector (PAMPC version)

    :param v: 3D numpy vector or CasADi MX
    :return: the corresponding skew-symmetric matrix of v with the same data type as v
    """

    if isinstance(v, np.ndarray):
        return np.array([[0, -v[0], -v[1], -v[2]],
                         [v[0], 0, v[2], -v[1]],
                         [v[1], -v[2], 0, v[0]],
                         [v[2], v[1], -v[0], 0]])

    return cs.vertcat(
        cs.horzcat(0, -v[0], -v[1], -v[2]),
        cs.horzcat(v[0], 0, v[2], -v[1]),
        cs.horzcat(v[1], -v[2], 0, v[0]),
        cs.horzcat(v[2], v[1], -v[0], 0))

def v_dot_q(v, q):
    rot_mat = q_to_rot_mat(q)
    if isinstance(q, np.ndarray):
        return rot_mat.dot(v)

    return cs.mtimes(rot_mat, v)


def q_to_rot_mat(q):
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    if isinstance(q, np.ndarray):
        rot_mat = np.array([
            [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]])

    else:
        rot_mat = cs.vertcat(
            cs.horzcat(1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)),
            cs.horzcat(2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)),
            cs.horzcat(2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)))

    return rot_mat

def quaternion_inverse(q):
    w, x, y, z = q[0], q[1], q[2], q[3]

    if isinstance(q, np.ndarray):
        return np.array([w, -x, -y, -z])
    else:
        return cs.vertcat(w, -x, -y, -z)

def safe_mkdir_recursive(directory, overwrite=False):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(directory):
                pass
            else:
                raise
    else:
        if overwrite:
            try:
                shutil.rmtree(directory)
            except:
                print('Error while removing directory: {0}'.format(directory))

class Quad3DOptimizer:
    def __init__(self, quad, t_horizon=1, n_nodes=20,
                 q_cost=None, r_cost=None, q_mask=None,
                 model_name="drone_acados_mpc", solver_options=None):
        """
        :param quad: quadrotor object
        :type quad: Quadrotor3D
        :param t_horizon: time horizon for MPC optimization
        :param n_nodes: number of optimization nodes until time horizon
        :param q_cost: diagonal of Q matrix for LQR cost of MPC cost function. Must be a numpy array of length 12.
        :param r_cost: diagonal of R matrix for LQR cost of MPC cost function. Must be a numpy array of length 4.
        :param B_x: dictionary of matrices that maps the outputs of the gp regressors to the state space.
        :param gp_regressors: Gaussian Process ensemble for correcting the nominal model
        :type gp_regressors: GPEnsemble
        :param q_mask: Optional boolean mask that determines which variables from the state compute towards the cost
        function. In case no argument is passed, all variables are weighted.
        :param solver_options: Optional set of extra options dictionary for solvers.
        :param rdrv_d_mat: 3x3 matrix that corrects the drag with a linear model according to Faessler et al. 2018. None
        if not used
        """

        # Weighted squared error loss function q = (p_xyz, q_xyz, v_xyz, w_xyz), r = (u1, u2, u3, u4)
        if q_cost is None:
            q_cost = np.array([1, 1, 1, 0.5, 0.5, 0.5, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
        if r_cost is None:
            r_cost = np.array([0.45, 0.45, 0.45, 0.45])

        self.T = t_horizon  # Time horizon
        self.N = n_nodes  # number of control nodes within horizon

        self.quad = quad

        self.max_u = quad.max_input_value
        self.min_u = quad.min_input_value

        # Declare model variables
        self.p = cs.MX.sym('p', 3)  # position
        self.q = cs.MX.sym('q', 4)  # quaternion (wxyz)
        self.v = cs.MX.sym('v', 3)  # velocity
        self.w = cs.MX.sym('w', 3)  # angle rate

        # Full state vector (13-dimensional)
        self.x = cs.vertcat(self.p, self.q, self.v, self.w)
        self.state_dim = 13

        # Control input vector
        u1 = cs.MX.sym('u1')
        u2 = cs.MX.sym('u2')
        u3 = cs.MX.sym('u3')
        u4 = cs.MX.sym('u4')
        self.u = cs.vertcat(u1, u2, u3, u4)

        # Nominal model equations symbolic function (no GP)
        self.quad_xdot_nominal = self.quad_dynamics()

        # Linearized model dynamics symbolic function
        self.quad_xdot_jac = self.linearized_quad_dynamics()

        # Build full model. Will have 13 variables. self.dyn_x contains the symbolic variable that
        # should be used to evaluate the dynamics function. It corresponds to self.x
        acados_models, _ = self.acados_setup_model(
            self.quad_xdot_nominal(x=self.x, u=self.u)['x_dot'], model_name)

        # ### Setup and compile Acados OCP solvers ### #
        self.acados_ocp_solver = {}

        # Add one more weight to the rotation (use quaternion norm weighting in acados)
        q_diagonal = np.concatenate((q_cost[:3], np.mean(q_cost[3:6])[np.newaxis], q_cost[3:]))
        if q_mask is not None:
            q_mask = np.concatenate((q_mask[:3], np.zeros(1), q_mask[3:]))
            q_diagonal *= q_mask

        # Ensure current working directory is current folder
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        self.acados_models_dir = '../../acados_models'
        safe_mkdir_recursive(os.path.join(os.getcwd(), self.acados_models_dir))

        for key, key_model in zip(acados_models.keys(), acados_models.values()):
            nx = key_model.x.size()[0]
            nu = key_model.u.size()[0]
            ny = nx + nu
            n_param = key_model.p.size()[0] if isinstance(key_model.p, cs.MX) else 0

            acados_source_path = os.environ['ACADOS_SOURCE_DIR']
            sys.path.insert(0, '../common')

            # Create OCP object to formulate the optimization
            ocp = AcadosOcp()
            ocp.acados_include_path = acados_source_path + '/include'
            ocp.acados_lib_path = acados_source_path + '/lib'
            ocp.model = key_model
            ocp.dims.N = self.N
            ocp.solver_options.tf = t_horizon

            # Initialize parameters
            ocp.dims.np = n_param
            ocp.parameter_values = np.zeros(n_param)

            ocp.cost.cost_type = 'LINEAR_LS'
            ocp.cost.cost_type_e = 'LINEAR_LS'

            ocp.cost.W = np.diag(np.concatenate((q_diagonal, r_cost)))
            ocp.cost.W_e = np.diag(q_diagonal)
            terminal_cost = 0 if solver_options is None or not solver_options["terminal_cost"] else 1
            ocp.cost.W_e *= terminal_cost

            ocp.cost.Vx = np.zeros((ny, nx))
            ocp.cost.Vx[:nx, :nx] = np.eye(nx)
            ocp.cost.Vu = np.zeros((ny, nu))
            ocp.cost.Vu[-4:, -4:] = np.eye(nu)

            ocp.cost.Vx_e = np.eye(nx)

            # Initial reference trajectory (will be overwritten)
            x_ref = np.zeros(nx)
            ocp.cost.yref = np.concatenate((x_ref, np.array([0.0, 0.0, 0.0, 0.0])))
            ocp.cost.yref_e = x_ref

            # Initial state (will be overwritten)
            ocp.constraints.x0 = x_ref

            # Set constraints
            ocp.constraints.lbu = np.array([self.min_u] * 4)
            ocp.constraints.ubu = np.array([self.max_u] * 4)
            ocp.constraints.idxbu = np.array([0, 1, 2, 3])

            # Solver options
            ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
            ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
            ocp.solver_options.integrator_type = 'ERK'
            ocp.solver_options.print_level = 0
            ocp.solver_options.nlp_solver_type = 'SQP_RTI' if solver_options is None else solver_options["solver_type"]

            # Compile acados OCP solver if necessary
            json_file = os.path.join(self.acados_models_dir, key_model.name + '_acados_ocp.json')
            self.acados_ocp_solver[key] = AcadosOcpSolver(ocp, json_file=json_file)

    def clear_acados_model(self):
        """
        Removes previous stored acados models to avoid name conflicts.
        """

        json_file = os.path.join(self.acados_models_dir, 'acados_ocp.json')
        if os.path.exists(json_file):
            os.remove(os.path.join(os.getcwd(), json_file))
        compiled_model_dir = os.path.join(os.getcwd(), 'c_generated_code')
        if os.path.exists(compiled_model_dir):
            shutil.rmtree(compiled_model_dir)

    def acados_setup_model(self, nominal, model_name):
        """
        Builds an Acados symbolic models using CasADi expressions.
        :param model_name: name for the acados model. Must be different from previously used names or there may be
        problems loading the right model.
        :param nominal: CasADi symbolic nominal model of the quadrotor: f(self.x, self.u) = x_dot, dimensions 13x1.
        :return: Returns a total of three outputs, where m is the number of GP's in the GP ensemble, or 1 if no GP:
            - A dictionary of m AcadosModel of the GP-augmented quadrotor
            - A dictionary of m CasADi symbolic nominal dynamics equations with GP mean value augmentations (if with GP)
        :rtype: dict, dict, cs.MX
        """

        def fill_in_acados_model(x, u, p, dynamics, name):

            x_dot = cs.MX.sym('x_dot', dynamics.shape)
            f_impl = x_dot - dynamics

            # Dynamics model
            model = AcadosModel()
            model.f_expl_expr = dynamics
            model.f_impl_expr = f_impl
            model.x = x
            model.xdot = x_dot
            model.u = u
            model.p = p
            model.name = name

            return model

        acados_models = {}
        dynamics_equations = {}

        dynamics_equations[0] = nominal

        x_ = self.x
        dynamics_ = nominal

        acados_models[0] = fill_in_acados_model(x=x_, u=self.u, p=[], dynamics=dynamics_, name=model_name)

        return acados_models, dynamics_equations

    def quad_dynamics(self):
        """
        Symbolic dynamics of the 3D quadrotor model. The state consists on: [p_xyz, a_wxyz, v_xyz, r_xyz]^T, where p
        stands for position, a for angle (in quaternion form), v for velocity and r for body rate. The input of the
        system is: [u_1, u_2, u_3, u_4], i.e. the activation of the four thrusters.

        :param rdrv_d: a 3x3 diagonal matrix containing the D matrix coefficients for a linear drag model as proposed
        by Faessler et al.

        :return: CasADi function that computes the analytical differential state dynamics of the quadrotor model.
        Inputs: 'x' state of quadrotor (6x1) and 'u' control input (2x1). Output: differential state vector 'x_dot'
        (6x1)
        """

        x_dot = cs.vertcat(self.p_dynamics(), self.q_dynamics(), self.v_dynamics(), self.w_dynamics())
        return cs.Function('x_dot', [self.x[:13], self.u], [x_dot], ['x', 'u'], ['x_dot'])

    def p_dynamics(self):
        return self.v

    def q_dynamics(self):
        return 1 / 2 * cs.mtimes(skew_symmetric(self.w), self.q)

    def v_dynamics(self):
        """
        :param rdrv_d: a 3x3 diagonal matrix containing the D matrix coefficients for a linear drag model as proposed
        by Faessler et al. None, if no linear compensation is to be used.
        """

        f_thrust = self.u
        g = cs.vertcat(0.0, 0.0, 9.81)
        a_thrust = cs.vertcat(0.0, 0.0, f_thrust[0] + f_thrust[1] + f_thrust[2] + f_thrust[3]) / self.quad.mass

        v_dynamics = v_dot_q(a_thrust, self.q) - g

        return v_dynamics

    def w_dynamics(self):
        f_thrust = self.u

        y_f = cs.MX(self.quad.y_f)
        x_f = cs.MX(self.quad.x_f)
        c_f = cs.MX(self.quad.z_l_tau)
        return cs.vertcat(
            (cs.mtimes(f_thrust.T, y_f) + (self.quad.J[1] - self.quad.J[2]) * self.w[1] * self.w[2]) / self.quad.J[0],
            (cs.mtimes(f_thrust.T, x_f) + (self.quad.J[2] - self.quad.J[0]) * self.w[2] * self.w[0]) / self.quad.J[1],
            (cs.mtimes(f_thrust.T, c_f) + (self.quad.J[0] - self.quad.J[1]) * self.w[0] * self.w[1]) / self.quad.J[2])

    def linearized_quad_dynamics(self):
        """
        Jacobian J matrix of the linearized dynamics specified in the function quad_dynamics. J[i, j] corresponds to
        the partial derivative of f_i(x) wrt x(j).

        :return: a CasADi symbolic function that calculates the 13 x 13 Jacobian matrix of the linearized simplified
        quadrotor dynamics
        """

        jac = cs.MX(self.state_dim, self.state_dim)

        # Position derivatives
        jac[0:3, 7:10] = cs.diag(cs.MX.ones(3))

        # Angle derivatives
        jac[3:7, 3:7] = skew_symmetric(self.w) / 2
        jac[3, 10:] = 1 / 2 * cs.horzcat(-self.q[1], -self.q[2], -self.q[3])
        jac[4, 10:] = 1 / 2 * cs.horzcat(self.q[0], -self.q[3], self.q[2])
        jac[5, 10:] = 1 / 2 * cs.horzcat(self.q[3], self.q[0], -self.q[1])
        jac[6, 10:] = 1 / 2 * cs.horzcat(-self.q[2], self.q[1], self.q[0])

        # Velocity derivatives
        a_u = (self.u[0] + self.u[1] + self.u[2] + self.u[3]) / self.quad.mass
        jac[7, 3:7] = 2 * cs.horzcat(a_u * self.q[2], a_u * self.q[3], a_u * self.q[0], a_u * self.q[1])
        jac[8, 3:7] = 2 * cs.horzcat(-a_u * self.q[1], -a_u * self.q[0], a_u * self.q[3], a_u * self.q[2])
        jac[9, 3:7] = 2 * cs.horzcat(0, -2 * a_u * self.q[1], -2 * a_u * self.q[1], 0)

        # Rate derivatives
        jac[10, 10:] = (self.quad.J[1] - self.quad.J[2]) / self.quad.J[0] * cs.horzcat(0, self.w[2], self.w[1])
        jac[11, 10:] = (self.quad.J[2] - self.quad.J[0]) / self.quad.J[1] * cs.horzcat(self.w[2], 0, self.w[0])
        jac[12, 10:] = (self.quad.J[0] - self.quad.J[1]) / self.quad.J[2] * cs.horzcat(self.w[1], self.w[0], 0)

        return cs.Function('J', [self.x, self.u], [jac])

    def set_reference_state(self, x_target=None, u_target=None, warm_start_option=2):
        """
        Sets the target state and pre-computes the integration dynamics with cost equations
        :param x_target: 13-dimensional target state (p_xyz, a_wxyz, v_xyz, r_xyz)
        :param u_target: 4-dimensional target control input vector (u_1, u_2, u_3, u_4)
        """

        if x_target is None:
            x_target = np.array([0, 0, 0,
                                 1, 0, 0, 0,
                                 0, 0, 0,
                                 0, 0, 0])
        if u_target is None:
            u_target = np.array([0, 0, 0, 0])

        # Set new target state
        # self.target = copy(x_target)

        ref = np.concatenate((x_target, u_target))

        for j in range(self.N):
            self.acados_ocp_solver[0].set(j, "yref", ref)

            if warm_start_option == 1:
                self.acados_ocp_solver[0].set(j, "x", x_target)  # initial guess
                self.acados_ocp_solver[0].set(j, "u", u_target)  # initial guess
            elif warm_start_option == 2:
                # Use the last solution as the initial guess
                if hasattr(self, 'last_solution_x') and hasattr(self, 'last_solution_u'):
                    self.acados_ocp_solver[0].set(j, "x", self.last_solution_x[j, :])
                    self.acados_ocp_solver[0].set(j, "u", self.last_solution_u[j, :])
                else:
                    self.acados_ocp_solver[0].set(j, "x", x_target)
                    self.acados_ocp_solver[0].set(j, "u", u_target)

        self.acados_ocp_solver[0].set(self.N, "yref", ref[:-4])

        if warm_start_option == 1:
            self.acados_ocp_solver[0].set(self.N, "x", x_target)
        elif warm_start_option == 2:
            # Use the last solution as the initial guess
            if hasattr(self, 'last_solution_x') and hasattr(self, 'last_solution_u'):
                self.acados_ocp_solver[0].set(self.N, "x", self.last_solution_x[self.N, :])
            else:
                self.acados_ocp_solver[0].set(self.N, "x", x_target)

        return 0

    def set_reference_trajectory(self, x_target, u_target, warm_start_option=1):
        """
        Sets the reference trajectory and pre-computes the cost equations for each point in the reference sequence.
        :param x_target: Nx13-dimensional reference trajectory (p_xyz, angle_wxyz, v_xyz, rate_xyz). It is passed in the
        form of a 4-length list, where the first element is a Nx3 numpy array referring to the position targets, the
        second is a Nx4 array referring to the quaternion, two more Nx3 arrays for the velocity and body rate targets.
        :param u_target: Nx4-dimensional target control input vector (u1, u2, u3, u4)
        """

        if u_target is not None:
            assert x_target.shape[0] == (u_target.shape[0] + 1) or x_target.shape[0] == u_target.shape[0]

        # If not enough states in target sequence, append last state until required length is met
        while x_target.shape[0] < self.N + 1:
            x_target = np.concatenate((x_target, np.expand_dims(x_target[-1, :], 0)), 0)
            if u_target is not None:
                u_target = np.concatenate((u_target, np.expand_dims(u_target[-1, :], 0)), 0)

        # self.target = copy(x_target) 
        # for j in range(self.N):
        #     ref = np.concatenate((x_target[j, :], u_target[j, :]))
        #     self.acados_ocp_solver[0].set(j, "yref", ref)
        #     self.acados_ocp_solver[0].set(j, "x", x_target[j, :])# initial guess
        #     self.acados_ocp_solver[0].set(j, "u", u_target[j, :])# initial guess

        for j in range(self.N):
            ref = np.concatenate((x_target[j, :], u_target[j, :]))
            self.acados_ocp_solver[0].set(j, "yref", ref)
            if warm_start_option == 1:
                self.acados_ocp_solver[0].set(j, "x", x_target[j, :])  # initial guess
                self.acados_ocp_solver[0].set(j, "u", u_target[j, :])  # initial guess
            elif warm_start_option == 2:
                # Use the last solution as the initial guess
                if hasattr(self, 'last_solution_x') and hasattr(self, 'last_solution_u'):
                    self.acados_ocp_solver[0].set(j, "x", self.last_solution_x[j, :])
                    self.acados_ocp_solver[0].set(j, "u", self.last_solution_u[j, :])
                else:
                    self.acados_ocp_solver[0].set(j, "x", x_target[j, :])
                    self.acados_ocp_solver[0].set(j, "u", u_target[j, :])

        # the last MPC node has only a state reference but no input reference
        self.acados_ocp_solver[0].set(self.N, "yref", x_target[self.N, :])
        if warm_start_option == 1:
            self.acados_ocp_solver[0].set(self.N, "x", x_target[self.N, :])
        elif warm_start_option == 2:
            # Use the last solution as the initial guess
            if hasattr(self, 'last_solution_x') and hasattr(self, 'last_solution_u'):
                self.acados_ocp_solver[0].set(self.N, "x", self.last_solution_x[self.N, :])
            else:
                self.acados_ocp_solver[0].set(self.N, "x", x_target[self.N, :])

        # self.acados_ocp_solver[0].set(self.N, "x", x_target[self.N, :])# initial guess
        return 0

    def run_optimization(self, initial_state=None, return_x=False):
        """
        Optimizes a trajectory to reach the pre-set target state, starting from the input initial state, that minimizes
        the quadratic cost function and respects the constraints of the system

        :param initial_state: 13-element list of the initial state. If None, 0 state will be used
        :param use_model: integer, select which model to use from the available options.
        :param return_x: bool, whether to also return the optimized sequence of states alongside with the controls.
        :param gp_regression_state: 13-element list of state for GP prediction. If None, initial_state will be used.
        :return: optimized control input sequence (flattened)
        """

        if initial_state is None:
            initial_state = np.array([0, 0, 0] + [1, 0, 0, 0] + [0, 0, 0] + [0, 0, 0])

        # Set initial state. Add gp state if needed
        x_init = initial_state
        # x_init = np.stack(x_init)

        # Set initial condition, equality constraint
        self.acados_ocp_solver[0].set(0, 'lbx', x_init)
        self.acados_ocp_solver[0].set(0, 'ubx', x_init)

        # Solve OCP
        self.acados_ocp_solver[0].solve()

        # Get u
        u_opt_acados = np.ndarray((self.N, 4))
        x_opt_acados = np.ndarray((self.N + 1, len(x_init)))
        x_opt_acados[0, :] = self.acados_ocp_solver[0].get(0, "x")
        for i in range(self.N):
            u_opt_acados[i, :] = self.acados_ocp_solver[0].get(i, "u")
            x_opt_acados[i + 1, :] = self.acados_ocp_solver[0].get(i + 1, "x")

        # Save the last solution for warm starting
        self.last_solution_x = x_opt_acados
        self.last_solution_u = u_opt_acados

        u_opt_acados = np.reshape(u_opt_acados, (-1))
        return u_opt_acados if not return_x else (u_opt_acados, x_opt_acados)


class QuadrotorParam:

    def __init__(self, config):
        

        self.mass = config["mass"]
        self.max_input_value = config["max_total_t"]/4
        self.min_input_value = config["min_total_t"]/4
        self.l = config["l"]
        self.J = config["J"]
        self.g = config["g"]
        self.kt = config["kt"]
        self.km = config["km"]
        self.drone_frame = config["drone_frame"]
        self.freq = config["control_hz"]

        self.max_total_t = config["max_total_t"]

        self.m_T = np.array([[0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [1, 1, 1, 1]])

        if self.drone_frame=="+":
            arm_torque = self.l
            self.m_tau = np.array([[0, arm_torque, 0, -arm_torque],
                                [-arm_torque, 0, arm_torque, 0],
                                [self.km, -self.km, self.km, -self.km]])
            
        if self.drone_frame=="x":
            arm_torque = self.l/np.sqrt(2)
            self.m_tau = np.array([[arm_torque, -arm_torque, -arm_torque, arm_torque],
                            [-arm_torque, -arm_torque, arm_torque, arm_torque],
                            [self.km, -self.km, self.km, -self.km]])
            
        if self.drone_frame=="mavic":
            arm_q_r_x=config["arm_q_r_x"]
            arm_q_r_y=config["arm_q_r_y"]

            arm_q_f_x=config["arm_q_f_x"]
            arm_q_f_y=config["arm_q_f_y"]
            self.m_tau = np.array([[-arm_q_f_y, arm_q_f_y, -arm_q_r_y, arm_q_r_y],
                            [-arm_q_f_x, -arm_q_f_x, arm_q_r_x, arm_q_r_x],
                            [self.km, -self.km, -self.km, self.km]])

        self.x_f = self.m_tau[1,:]
        self.y_f = self.m_tau[0,:]
        self.z_l_tau = self.m_tau[2,:]