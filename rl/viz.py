import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import nengo

from .utils import softmax


def rend(env):
    return plt.imshow(env.render())

def save_gifs(figure, images, trial, directory):
    ani = animation.ArtistAnimation(figure, images, interval=50, blit=True,
                                    repeat_delay=1e5)
    writergif = animation.PillowWriter(fps=10)
    ani.save(directory+'trial%s.gif' % (trial), writer=writergif)

def get_ac_output(net, env, n_grid=30, n_ranges=(6,6,4)):
    if "OneHot" in str(net.representation):
        x = n_ranges[0]
        y = n_ranges[1]
        z = n_ranges[2]
        X, Y, Z = np.meshgrid(np.arange(x), np.arange(y), np.arange(z))
    else:
        x = np.linspace(0, env.width, n_grid)
        y = np.linspace(0, env.height, n_grid)
        z = np.linspace(0, 360, n_grid)
        X, Y, Z = np.meshgrid(x, y, z)

    pts = np.array([X, Y, Z]).reshape(3, -1)
    SSPs = [net.representation.map(x).copy() for x in pts.T]

    if net.state_neurons is not None:
        _, A = nengo.utils.ensemble.tuning_curves(net.ensemble, net.sim, inputs=SSPs)
        if "OneHot" in str(net.representation):
            A = A.reshape((x, y, z, -1))
        else:
            A = A.reshape((len(x), len(y), len(z), -1))
    else:
        try:
            A = np.array(SSPs).reshape((len(x), len(y), len(z), -1))
        except TypeError:
            A = np.array(SSPs).reshape((x, y, z, -1))

    w = net.sim.signals[net.sim.model.sig[net.rule.output]['_state_w']]
    V = A.dot(w.T)
    return softmax(V[:,:,:,1:], axis=-1), V[:,:,:,0], [X, Y, Z]

def plot_policy(policy, pts, values=None, plot_type='vector', ax=None, vmin='auto', vmax='auto', cmap='viridis'):
    if ax is None:
        fig, (ax, cax) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [50, 1]})

    policy = np.mean(policy, axis=2)
    values = np.mean(values, axis=2)

    if vmin == 'auto':
        vmin = round(values.min(), 2)
    if vmax == 'auto':
        vmax = round(values.max(), 2)

    if values is not None:
        im = ax.contourf(pts[0][:,:,0], pts[1][:,:,0],
                         values, levels=np.linspace(vmin, vmax, 10),
                         vmin=vmin, vmax=vmax, cmap=cmap)
        ticks = [vmin, round((vmax+vmin)/2, 2), vmax]
        cbar = fig.colorbar(im, cax=cax, ticks=ticks, orientation='vertical')
        cbar.set_label('Value', va='top', ha='left', rotation=90, in_layout=True)
        cbar.ax.set_yticklabels(ticks)

    if plot_type == 'vector':
        ax.quiver(pts[0][:,:,0], pts[1][:,:,0],
                  policy[:,:,2]-policy[:,:,3],
                  policy[:,:,0]-policy[:,:,1], color='k')
    elif plot_type == 'stream':
        ax.streamplot(pts[0][:,:,0], pts[1][:,:,0],
                      policy[:,:,2]-policy[:,:,3],
                      policy[:,:,1]-policy[:,:,0], color='k')
    return ax

def plot_table(P):
    plt.figure(figsize=(14, 14))
    for i in range(4):
        for j in range(4):
            plt.subplot(4, 4, i*4+j+1)
            plt.imshow(P[:,:,j,i], vmin=0, vmax=1)
            plt.colorbar()
            if j == 0:
                plt.ylabel(['value', 'turn left', 'turn right', 'forward'][i])
            if i == 0:
                plt.title(['east', 'south', 'west', 'north'][j])
    plt.show()
