import numpy as np
import numpy.random as npr
import time
import itertools
import os
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.ops import index, index_add, index_update
from jax.experimental import optimizers
from jax import lax
from jax.nn import sigmoid
from jax.nn.initializers import glorot_normal, normal
from functools import partial
from jax import value_and_grad
from jax import random

def init_FC_params(scale, layer_sizes, rng=npr.RandomState(0)):
    return [(scale * rng.randn(m, n), scale * rng.randn(n))
            for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])]

def init_GRU_params(rng, input_shape, W_init=glorot_normal(), b_init=normal()):
    """ Initialize the GRU layer """
    batch_size, hiden_dim, input_data_dim = input_shape   #input_data_dim=X,t
    # H0 = jnp.zeros(hiden_dim)
    H0 = b_init(rng, (hiden_dim,))

    k1, k2, k3 = random.split(rng, num=3)
    # W takes the X data and U takes the previous hidden state,
    # then combined by adding together with the bias post the matrix dot
    reset_W, reset_U, reset_b = (
        W_init(k1, (input_data_dim, hiden_dim)),
        W_init(k2, (hiden_dim, hiden_dim)),
        b_init(k3, (hiden_dim,)),)

    k1, k2, k3 = random.split(rng, num=3)
    update_W, update_U, update_b = (
        W_init(k1, (input_data_dim, hiden_dim)),
        W_init(k2, (hiden_dim, hiden_dim)),
        b_init(k3, (hiden_dim,)),)

    k1, k2, k3 = random.split(rng, num=3)
    out_W, out_U, out_b = (
        W_init(k1, (input_data_dim, hiden_dim)),
        W_init(k2, (hiden_dim, hiden_dim)),
        b_init(k3, (hiden_dim,)),)

    GRU_params = ((update_W, update_U, update_b), (reset_W, reset_U, reset_b), (out_W, out_U, out_b))
    return H0, GRU_params

def relu(x):
    return jnp.maximum(0, x)

def forward(params, hidden):
    # input = jnp.concatenate((t, X), 0)  # M x D+1
    activations = hidden

    for w, b in params[:-1]:
        outputs = jnp.dot(activations, w) + b
        activations = relu(outputs)

    final_w, final_b = params[-1]
    u = jnp.dot(activations, final_w) + final_b
    return jnp.reshape(u, ())  # need scalar for grad
vforward = vmap(forward, in_axes=(None, 0))

def grad_forward(params, hidden):
    gradu = grad(forward, argnums=(1))  # <wrt H
    Du = gradu(params, hidden)
    return Du
vgrad_forward = vmap(grad_forward, in_axes=(None, 0))

def GRU_forward(GRU_params, hidden, t, X):
    inp = jnp.concatenate((t, X), 0)  # M x D+1
    (update_W, update_U, update_b), (reset_W, reset_U, reset_b), (
            out_W, out_U, out_b) = GRU_params

    reset_gate = sigmoid(jnp.dot(inp, reset_W) +
                         jnp.dot(hidden, reset_U) + reset_b)
    update_gate = sigmoid(jnp.dot(inp, update_W) +
                          jnp.dot(hidden, update_U) + update_b)
    output_gate = jnp.tanh(jnp.dot(inp, out_W)
                          + jnp.dot(jnp.multiply(reset_gate, hidden), out_U)
                          + out_b)
    output = jnp.multiply(update_gate, hidden) + jnp.multiply(1-update_gate, output_gate)

    return output
vGRU_forward = vmap(GRU_forward, in_axes=(None, 0, 0, 0))

def jac_GRU(GRU_params, hidden, t, X):
    jacGRU = jax.jacfwd(GRU_forward, argnums=(3))
    DGRU = jacGRU(GRU_params, hidden, t, X)
    return DGRU
vgrad_GRU = vmap(grad_forward, in_axes=(None, 0, 0, 0))

def fetch_minibatch(T, M, N, D):  # Generate time + a Brownian motion
    Dt = jnp.zeros((M, N, 1))  # M x (N+1) x 1
    DW = jnp.zeros((M, N, D))  # M x (N+1) x D
    dt = T / N
    new_Dt = index_update(Dt, index[:, :, :], dt)
    new_dw = jnp.sqrt(dt) * np.random.normal(size=(M, N, D))
    new_DW = index_update(DW, index[:, :, :], new_dw)
    return new_Dt, new_DW

@jit
def XYZpaths(params, t, W, X0):
    H0, GRU_params, FC_params = params
    t0 = jnp.array([0.])
    Y0 = jnp.asarray([forward(FC_params, H0)])
    Z0 = jnp.dot(grad_forward(FC_params, H0), jac_GRU(GRU_params, H0, t0, X0))

    Y0_tilde = jnp.asarray([0.])
    initial = (t0, X0, H0, Y0, Y0_tilde, Z0)

    def body(carry, tW):
        dt, dW = tW
        t0, X0, H0, Y0, Y0_tilde, Z0 = carry
        X1 = X0 + mu_tf(t0, X0, Y0, Z0) * (dt) + jnp.dot(sigma_tf(t0, X0, Y0), dW)
        t1 = t0 + dt

        H1 = GRU_forward(GRU_params, H0, t1, X1)
        Y1 = jnp.asarray([forward(FC_params, H1)])

        Z1 = jnp.dot(grad_forward(FC_params, H1), jac_GRU(GRU_params, H0, t1, X1))

        Y1_tilde = Y0 + phi_tf(t0, X0, Y0, Z0) * (dt) + jnp.dot(jnp.dot(Z0.T, sigma_tf(t0, X0, Y0)), dW)

        carry_new = t1, X1, H1, Y1, Y1_tilde, Z1
        return (carry_new, carry)

    final_state, trace = lax.scan(body, initial, (t, W))

    t_trace, X_trace, H_trace, Y_trace, Y_tilde_trace, Z_trace = trace
    t_end, X_end, H_end, Y_end, Y_tilde_end, Z_end = final_state
    X = jnp.concatenate((X_trace, jnp.reshape(X_end, (1, D))), 0)
    H = jnp.concatenate((H_trace, jnp.reshape(H_end, (1, Hidden_dim))), 0)
    Y = jnp.concatenate((Y_trace, jnp.reshape(Y_end, (1, 1))), 0)
    Y_tilde = jnp.concatenate((Y_tilde_trace, jnp.reshape(Y_tilde_end, (1, 1))), 0)
    Z = jnp.concatenate((Z_trace, jnp.reshape(Z_end, (1, D))), 0)

    return X, H, Y, Y_tilde, Z
vXYZpaths = jit(vmap(XYZpaths, in_axes=(None, 0, 0, None)))

# jit static arg nums (0,1,2,3,)
def loss_function(params, t, W, Xzero):
    X, H, Y, Y_tilde, Z = vXYZpaths(params, t, W, Xzero)
    loss = jnp.sum(jnp.power(Y[:, 1:-1, :] - Y_tilde[:, 1:-1, :], 2))
    loss += jnp.sum(jnp.power(Y[:, -1, :] - vg_tf(X[:, -1, :]), 2))
    loss += jnp.sum(jnp.power(Z[:, -1, :] - vDg_tf(X[:, -1, :]), 2))
    return loss

@jit
def update(itcount, opt_state, t, W, X0):
    params = get_params(opt_state)
    return opt_update(itcount, grad(loss_function, argnums=0)(params, t, W, X0), opt_state)

def phi_tf(t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
    return 0.05 * (Y - jnp.dot(X, Z))  # M x 1

def g_tf(X):  # M x D
    return jnp.sum(X ** 2, keepdims=True)
vg_tf = vmap(g_tf)

def Dg_tf(X):
    def g(X):
        return jnp.sum(X ** 2)
    gradg = grad(g)
    Dg = gradg(X)
    return Dg
vDg_tf = vmap(Dg_tf)

def mu_tf(t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
    return jnp.zeros(D)  # M x D

def sigma_tf(t, X, Y):  # M x 1, M x D, M x 1
    return 0.4 * jnp.diag(X)  # M x D x D

def u_exact(t, X):  # (N+1) x 1, (N+1) x D
    r = 0.05
    sigma_max = 0.4
    return jnp.exp((r + sigma_max ** 2) * (T - t)) * jnp.sum(X ** 2, 1, keepdims=True)  # (N+1) x 1

if __name__ == "__main__":
    from jax.lib import xla_bridge
    print(xla_bridge.get_backend().platform)

    tot = time.time()
    M = 98  # number of trajectories (batch size)
    N = 50  # number of time snapshots
    D = 100  # number of dimensions
    T = 1.0
    Hidden_dim = 50  # 100 #2
    N_Iter = 2000  # 10 #10
    # step_size_list = [0.001]
    step_size_list = [0.001, 0.001, 0.001, 0.0001, 0.0001, 0.0001, 0.00001, 0.00001, 0.000001, 0.000001]
    FC_layers = [Hidden_dim] + 4 * [256] + [1]  # [12, 256, 256, 256, 256, 1]

    if D == 1:
        Xzero = jnp.array([1.0])
    else:
        Xzero = jnp.array([1.0, 0.5] * int(D / 2))

    tot = time.time()

    training_loss = []
    iteration = []
    loss_temp = jnp.array([])

    previous_it = 0
    # Init params
    input_shape = M, Hidden_dim, D + 1
    # (H0, (update_W, update_U, update_b), (reset_W, reset_U, reset_b), (out_W, out_U, out_b))
    # Generate key which is used to generate jax random numbers
    key = random.PRNGKey(1)
    H0, GRU_params = init_GRU_params(key, input_shape)
    param_scale = 0.1
    FC_params = init_FC_params(param_scale, FC_layers)
    params = [H0, GRU_params, FC_params]

    for step_size in step_size_list:
        # Optimizer
        opt_init, opt_update, get_params = optimizers.adam(step_size)
        opt_state = opt_init(params)

        start_time = time.time()
        itercount = itertools.count()
        if iteration != []:
            previous_it = iteration[-1] + 10
        for it in range(previous_it, previous_it + N_Iter):
            t_batch, W_batch = fetch_minibatch(T, M, N, D)  # M x (N+1) x 1, M x (N+1) x D
            opt_state = update(next(itercount), opt_state, t_batch, W_batch, Xzero)
            params = get_params(opt_state)

            loss = loss_function(params, t_batch, W_batch, Xzero)
            loss_temp = jnp.append(loss_temp, loss)

            if it % 10 == 0:  # 100
                elapsed = time.time() - start_time
                print('It: %d, Loss: %.3e, Time: %.2f, Learning Rate: %.3e' %
                      (it, loss, elapsed, step_size))
                start_time = time.time()
                # Loss
                training_loss.append(loss_temp.mean())
                loss_temp = jnp.array([])
                iteration.append(it)

    graph = np.stack((iteration, training_loss))

    print("total time:", time.time() - tot, "s")

    t_test, W_test = fetch_minibatch(T, M, N, D)
    X_pred, H_pred, Y_pred, Y_tilde_pred, Z = vXYZpaths(params, t_test, W_test, Xzero)

    Dt = jnp.zeros((M, N + 1, 1))  # M x (N+1) x 1
    dt = T / N
    new_Dt = index_update(Dt, index[:, 1:, :], dt)
    t_plot = jnp.cumsum(new_Dt, axis=1)  # M x (N+1) x 1

    Y_test = jnp.reshape(u_exact(jnp.reshape(t_plot[0:M, :, :], [-1, 1]), jnp.reshape(X_pred[0:M, :, :], [-1, D])),
                         [M, -1, 1])

    np.save('t_test.npy', t_test)
    np.save('W_test.npy', W_test)
    np.save('t_plot.npy', t_plot)
    np.save('X_pred.npy', X_pred)
    np.save('Y_pred.npy', Y_pred)
    np.save('Y_tilde_pred.npy', Y_tilde_pred)
    np.save('Y_test.npy', Y_test)
    np.save('graph.npy', graph)

    from google.colab import files

    # files.download('t_test.npy')
    # files.download('W_test.npy')
    # files.download('t_plot.npy')
    # files.download('X_pred.npy')
    # files.download('Y_pred.npy')
    # files.download('Y_tilde_pred.npy')
    # files.download('Y_test.npy')
    # files.download('graph.npy')