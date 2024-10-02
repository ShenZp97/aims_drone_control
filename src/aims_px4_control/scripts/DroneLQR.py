
import numpy as np
# from scipy.integrate import solve_ivp
# from scipy.linalg import solve_continuous_are as care
from scipy.linalg import block_diag
# from scipy.linalg import solve_discrete_are as dare
from numpy.linalg import multi_dot
import control as ct
from time import time

# Quadrotor Parameters
# m = 0.5
# l = 0.175
# J = np.diag([0.0023, 0.0023, 0.004])
# g = 9.81
# kt = 1.0
# km = 0.0245
# drone_frame = '+'
# max_total_t = m*g/0.25

class LQRController:
    def __init__(self, config):

        self.m = config["mass"]
        self.l = config["l"]
        self.J = np.diag(config["J"])
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

        self.h = 1/self.freq  # Sampling time
        self.T = np.diag([1, -1, -1, -1])
        self.H = np.vstack((np.zeros((1, 3)), np.eye(3)))

        # x0 = np.hstack((np.array([0, 2, 5]), np.array([0, 0, 0, 1]), np.zeros(3), np.zeros(3)))
        x0 = np.hstack((np.array([0, 2, 5]), np.array([1, 0, 0, 0]), np.zeros(3), np.zeros(3)))
        self.uhover = (self.m * self.g / 4) * np.ones(4)
        q0 = np.array([1, 0, 0, 0])

        # t0 = time()
        A, B = self.linearize_dynamics(x0, self.uhover)
        # print("Linearization time:", time()-t0)
        A_tilde = multi_dot([self.E(q0).T, A, self.E(q0)])
        B_tilde = self.E(q0).T @ B

        # Q = 1.0*np.eye(12)
        # Q = np.diag([0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.01, 0.01, 0.01])
        Q = np.diag(config['Q'])
        # R = 0.1 * np.eye(4)
        R = np.diag(config['R'])
        t0 = time()
        self.K = self.dlqr(A_tilde, B_tilde, Q, R)
        print("LQR gain time:", time()-t0)

    # Quaternion Operations
    def hat(self, v):
        return np.array([[0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]])

    def L(self, q):
        s, v = q[0], q[1:4]
        return np.vstack((np.hstack((s, -v.T)),
                        np.hstack((np.expand_dims(v,1), s*np.eye(3) + self.hat(v)))))


    def qtoQ(self, q):
        return self.H.T @ self.T @ self.L(q) @ self.T @ self.L(q) @ self.H

    def G(self, q):
        return self.L(q) @ self.H

    def rptoq(self, phi):
        return np.hstack((1, phi)) / np.sqrt(1 + phi.T @ phi)

    def qtorp(self, q):
        return q[1:4]

# def block_diag_manual(*arrs):
#     """Construct a block diagonal matrix from the provided arrays."""
#     # Calculate the total size needed by summing shapes of arrays
#     shapes = np.array([a.shape for a in arrs])
#     size = np.sum(shapes, axis=0)
    
#     # Create an output array of zeros of the appropriate size
#     out = np.zeros(size, dtype=arrs[0].dtype)
    
#     # Fill the output array with input arrays at the correct offsets
#     r, c = 0, 0
#     for a in arrs:
#         nrows, ncols = a.shape
#         out[r:r+nrows, c:c+ncols] = a
#         r += nrows
#         c += ncols
    
#     return out

    def E(self, q):
        return block_diag(np.eye(3), self.G(q), np.eye(6))

    # Dynamics of the Quadrotor
    def quad_dynamics(self, t, x, u):
        r, q, v, omega = x[:3], x[3:7], x[7:10], x[10:13]
        q = q / np.linalg.norm(q)  # normalize quaternion
        Q = self.qtoQ(q)
        
        r_dot = Q @ v
        q_dot = 0.5 * self.L(q) @ self.H @ omega
        v_dot = Q.T @ np.array([0, 0, -self.g]) + (1/self.m) * self.m_T @ u - self.hat(omega) @ v
        omega_dot = np.linalg.inv(self.J) @ (-self.hat(omega) @ self.J @ omega + self.m_tau @ u)
        
        return np.hstack((r_dot, q_dot, v_dot, omega_dot))

    def quad_dynamics_rk4(self, x, u):
        # RK4 integrator
        k1 = self.quad_dynamics(0, x, u)
        k2 = self.quad_dynamics(0, x + 0.5 * self.h * k1, u)
        k3 = self.quad_dynamics(0, x + 0.5 * self.h * k2, u)
        k4 = self.quad_dynamics(0, x + self.h * k3, u)
        xn = x + (self.h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        xn[3:7] /= np.linalg.norm(xn[3:7])  # renormalize quaternion
        return xn


    def linearize_dynamics(self, x0, uhover):
        from scipy.optimize import approx_fprime
        A = approx_fprime(x0, lambda x: self.quad_dynamics_rk4(x, uhover), epsilon=1e-6)
        B = approx_fprime(uhover, lambda u: self.quad_dynamics_rk4(x0, u), epsilon=1e-6)
        return A, B

# LQR Controller
# def dlqr(A,B,Q,R):
#     """Solve the discrete time lqr controller.
    
#     x[k+1] = A x[k] + B u[k]
    
#     cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
#     """
#     #ref Bertsekas, p.151
    
#     #first, try to solve the ricatti equation
#     X = np.matrix(dare(A, B, Q, R))
    
#     #compute the LQR gain
#     K = np.matrix(np.linalg.inv(B.T*X*B+R)*(B.T*X*A))
    
#     eigVals, eigVecs = np.linalg.eig(A-B*K)
#     # return K, X, eigVals
#     return K


    def dlqr(self, A, B, Q, R):
        K,_,_ = ct.dlqr(A, B, Q, R)
        return K

    def controller_thrusts(self, x, x_des):
        q_des = x_des[3:7]
        q = x[3:7]

        # avoid quaternion flipping
        if np.dot(q, q_des) < 0.0:
            q_des = -q_des
        # if q[0]<0.0:
        #     q = -q
        # if q_des[0]<0.0:
        #     q_des = -q_des


        phi = self.qtorp(self.L(q_des).T @ q)
        delta_x = np.hstack((x[:3] - x_des[:3], phi, x[7:10] - x_des[7:10], x[10:13] - x_des[10:13]))
        
        u = self.uhover - self.K @ delta_x

        return u

    def controller_webotsrate(self, x, x_des):
        q_des = x_des[3:7]
        q = x[3:7]

        # avoid quaternion flipping
        if np.dot(q, q_des) < 0.0:
            q_des = -q_des
        # if q[0]<0.0:
        #     q = -q
        # if q_des[0]<0.0:
        #     q_des = -q_des

        phi = self.qtorp(self.L(q_des).T @ q)
        delta_x = np.hstack((x[:3] - x_des[:3], phi, x[7:10] - x_des[7:10], x[10:13] - x_des[10:13]))
        
        u = self.uhover - self.K @ delta_x

        x_next = self.quad_dynamics_rk4(x,u)

        return x_next[10:13], np.sum(u)

    def controller_px4rates(self, x, x_des):
        q_des = x_des[3:7]
        q = x[3:7]

        # avoid quaternion flipping
        if np.dot(q, q_des) < 0.0:
            q_des = -q_des
        # if q[0]<0.0:
        #     q = -q
        # if q_des[0]<0.0:
        #     q_des = -q_des

        phi = self.qtorp(self.L(q_des).T @ q)
        delta_x = np.hstack((x[:3] - x_des[:3], phi, x[7:10] - x_des[7:10], x[10:13] - x_des[10:13]))
        
        u = self.uhover - self.K @ delta_x

        x_next = self.quad_dynamics_rk4(x,u)

        return x_next[10:13], np.sum(u)/self.max_total_t

