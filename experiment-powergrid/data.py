# Generate simulation data
import autograd
import autograd.numpy as np

import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp

damping = 0.02

def hamiltonian_fn(coords):
    q, p = np.split(coords,2)
    H = -0.2 * np.cos(q) + 1.25 * p**2 - 0.1 * q # V1 = V2 = 1, B12 = 0.2, P1 = 0.1, m1 = 0.4
    return H

def dynamics_fn(t, coords):
    dcoords = autograd.grad(hamiltonian_fn)(coords)
    dqdt, dpdt = np.split(dcoords,2)
    S = np.concatenate([dpdt, -dqdt - damping * dpdt], axis=-1)
    return S

def fun_stage1(t, x):
      q, p = np.split(x,2)
      dqdt = 2.5 * p
      dpdt = -damping * 2.5 * p + 0.1 
      return np.concatenate([dqdt, dpdt], axis=-1)

def get_trajectory_stage1(t_span=[0,10], timescale=10, y0=None, noise_std=0.0):
    t_eval = np.linspace(t_span[0], t_span[1], int(timescale*(t_span[1]-t_span[0])))
    stage1_ivp = solve_ivp(fun=fun_stage1, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10)
    q_stage1, p_stage1 = stage1_ivp['y']
    return q_stage1, p_stage1, t_eval

def get_y0():
    t_fault = np.random.rand() * 2   + 0.1
    y_stable = np.random.rand()
    x1, y1, t1 = get_trajectory_stage1(y0=np.array([y_stable, 0]),t_span=[0,t_fault],timescale = 100, noise_std=0)
    return np.array([x1[-1],y1[-1]])


def get_trajectory(t_span=[0,10], timescale=10, radius=None, y0=None, noise_std=0.05, **kwargs):
    t_eval = np.linspace(t_span[0], t_span[1], int(timescale*(t_span[1]-t_span[0])))
    
    # get initial state
    if y0 is None:
        y0 = get_y0()
        # q = np.random.rand() * 3.14
        # p = 0
        # y0 = np.array([q, p])

    spring_ivp = solve_ivp(fun=dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10, **kwargs)
    q, p = spring_ivp['y'][0], spring_ivp['y'][1]
    dydt = [dynamics_fn(None, y) for y in spring_ivp['y'].T]
    dydt = np.stack(dydt).T
    dqdt, dpdt = np.split(dydt,2)
    
    # add noise
    q += np.random.randn(*q.shape)*noise_std
    # dqdt += np.random.randn(*dqdt.shape)*noise_std
    p += np.random.randn(*p.shape)*noise_std * 0.2
    # dpdt += np.random.randn(*dpdt.shape)*noise_std * 0.2
    return q, p, dqdt, dpdt, t_eval


def get_dataset(seed=0, samples=50, test_split=0.8, **kwargs):
    data = {'meta': locals()}

    # randomly sample inputs
    np.random.seed(seed)
    xs, dxs = [], []
    for s in range(samples):
        x, y, dx, dy, t = get_trajectory(**kwargs)
        xs.append( np.stack( [x, y]).T )
        dxs.append( np.stack( [dx, dy]).T )
        
    data['x'] = np.concatenate(xs)
    data['dx'] = np.concatenate(dxs).squeeze()

    # make a train/test split
    split_ix = int(len(data['x']) * test_split)
    split_data = {}
    for k in ['x', 'dx']:
        split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
    data = split_data
    return data

def get_field(xmin=-1.2, xmax=1.2, ymin=-1.2, ymax=1.2, gridsize=20):
    field = {'meta': locals()}

    # meshgrid to get vector field
    b, a = np.meshgrid(np.linspace(xmin, xmax, gridsize), np.linspace(ymin, ymax, gridsize))
    ys = np.stack([b.flatten(), a.flatten()])
    
    # get vector directions
    dydt = [dynamics_fn(None, y) for y in ys.T]
    dydt = np.stack(dydt).T

    field['x'] = ys.T
    field['dx'] = dydt.T
    return field