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

def init_random_params(scale, layer_sizes, rng=npr.RandomState(0)):
    return [(scale * rng.randn(m, n), scale * rng.randn(n))
            for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])]

def relu(x):
    return jnp.maximum(0, x)

def forward(params, t, X):
    input = jnp.concatenate((t, X), 0)  # M x D+1
    activations = input

    for w, b in params[:-1]:
        outputs = jnp.dot(activations, w) + b
        activations = jnp.tanh(outputs) #jnp.sin(outputs) #relu(outputs) #

    final_w, final_b = params[-1]
    u = jnp.dot(activations, final_w) + final_b
    return jnp.reshape(u, ())  # need scalar for grad
vforward = vmap(forward, in_axes=(None, 0, 0))

def grad_forward(params, t, X):
    gradu = grad(forward, argnums=(2))  # <wrt X only not params or t
    Du = gradu(params, t, X)
    return Du
vgrad_forward = vmap(grad_forward, in_axes=(None, 0, 0))

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
    t0 = jnp.array([0.])
    Y0 = jnp.asarray([forward(params, t0, X0)])
    Z0 = grad_forward(params, t0, X0)
    Y0_tilde = jnp.asarray([0.])
    initial = (t0, X0, Y0, Y0_tilde, Z0)
    def body(carry, tW):
        dt, dW = tW
        t0, X0, Y0, Y0_tilde, Z0 = carry
        X1 = X0 + mu_tf(t0, X0, Y0, Z0) * (dt) + jnp.dot(sigma_tf(t0, X0, Y0), dW)
        Y1_tilde = Y0 + phi_tf(t0, X0, Y0, Z0) * (dt) + jnp.dot(jnp.dot(Z0.T, sigma_tf(t0, X0, Y0)), dW)
        t1 = t0 + dt
        Y1 = jnp.asarray([forward(params, t1, X1)])
        Z1 = grad_forward(params, t1, X1)
        carry_new = t1, X1, Y1, Y1_tilde, Z1
        return (carry_new, carry)

    final_state, trace = lax.scan(body, initial, (t, W))
    t_trace, X_trace, Y_trace, Y_tilde_trace, Z_trace = trace
    t_end, X_end, Y_end, Y_tilde_end, Z_end = final_state
    X = jnp.concatenate((X_trace, jnp.reshape(X_end, (1, D))), 0)
    Y = jnp.concatenate((Y_trace, jnp.reshape(Y_end, (1, 1))), 0)
    Y_tilde = jnp.concatenate((Y_tilde_trace, jnp.reshape(Y_tilde_end, (1, 1))), 0)
    Z = jnp.concatenate((Z_trace, jnp.reshape(Z_end, (1, D))), 0)

    return X, Y, Y_tilde, Z
vXYZpaths = jit(vmap(XYZpaths, in_axes=(None, 0, 0, None)))

def loss_function(params, t, W, Xzero):
    alpha = 1
    beta = 1
    X, Y, Y_tilde, Z = vXYZpaths(params, t, W, Xzero)
    loss = jnp.sum(jnp.power(Y[:, 1:-1, :] - Y_tilde[:, 1:-1, :], 2))
    loss += alpha * jnp.sum(jnp.power(Y[:, -1, :] - vg_tf(X[:, -1, :]), 2))
    loss += beta * jnp.sum(jnp.power(Z[:, -1, :] - vDg_tf(X[:, -1, :]), 2))
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
# vsigma_tf = vmap(sigma_tf, in_axes=0)

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
    D = 100  # 50#100  # number of dimensions
    T = 1.0
    N_Iter = 2000  # 20
    # step_size_list = [0.001]
    step_size_list = [0.001, 0.001, 0.001, 0.0001, 0.0001, 0.0001, 0.00001, 0.00001, 0.000001, 0.000001]
    layers = [D + 1] + 4 * [256] + [1]  # [101, 256, 256, 256, 256, 1]
    param_scale = 0.1

    if D == 1:
        Xzero = jnp.array([1.0])
    else:
        Xzero = jnp.array([1.0, 0.5] * int(D / 2))

    tot = time.time()

    training_loss = []
    iteration = []
    loss_temp = jnp.array([])

    previous_it = 0
    # Optimizer
    params = init_random_params(param_scale, layers)

    for step_size in step_size_list:
        opt_init, opt_update, get_params = optimizers.adam(step_size)
        opt_state = opt_init(params)

        start_time = time.time()
        itercount = itertools.count()
        if iteration != []:
            previous_it = iteration[-1]+10
        for it in range(previous_it, previous_it + N_Iter):
            t_batch, W_batch = fetch_minibatch(T, M, N, D)  # M x (N+1) x 1, M x (N+1) x D

            opt_state = update(next(itercount), opt_state, t_batch, W_batch, Xzero)
            #perturb init X0
            # rng = npr.RandomState(0)
            # pertXzero = (1 + 0.01 * rng.randn(D)) * Xzero
            # opt_state = update(next(itercount), opt_state, t_batch, W_batch, pertXzero)

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
            #saving interim training params
            # if it in[200, 2000, 5000, 20000]:
            #     save_params = get_params(opt_state)
            #     np.save(f"{it}_params.npy", save_params)
            #     from google.colab import files
            #     files.download(f"{it}_params.npy")

    graph = np.stack((iteration, training_loss))

    print("total time:", time.time() - tot, "s")

    # np.random.seed(42)
    t_test, W_test = fetch_minibatch(T, M, N, D)
    X_pred, Y_pred, Y_tilde_pred, Z = vXYZpaths(params, t_test, W_test, Xzero)

    Dt = jnp.zeros((M, N + 1, 1))  # M x (N+1) x 1
    dt = T / N
    new_Dt = index_update(Dt, index[:, 1:, :], dt)
    t_plot = jnp.cumsum(new_Dt, axis=1)  # M x (N+1) x 1

    Y_test = jnp.reshape(u_exact(np.reshape(t_plot[0:M, :, :], [-1, 1]), jnp.reshape(X_pred[0:M, :, :], [-1, D])),
                         [M, -1, 1])

    np.save('t_test.npy', t_test)
    np.save('W_test.npy', W_test)
    np.save('t_plot.npy', t_plot)
    np.save('X_pred.npy', X_pred)
    np.save('Y_pred.npy', Y_pred)
    np.save('Y_tilde_pred.npy', Y_tilde_pred)
    np.save('Y_test.npy', Y_test)
    np.save('graph.npy', graph)

    # from google.colab import files
    # files.download('t_test.npy')
    # files.download('W_test.npy')
    # files.download('t_plot.npy')
    # files.download('X_pred.npy')
    # files.download('Y_pred.npy')
    # files.download('Y_tilde_pred.npy')
    # files.download('Y_test.npy')
    # files.download('graph.npy')