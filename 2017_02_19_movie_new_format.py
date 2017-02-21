from functions import *
from plot_on_movie_2014_12_16 import *
import os
import time
from subprocess import call
import cine
import sys
import h5py


def get_frame_points_and_time(data, keys, i):
    pts = []
    for key in keys:
        if key != 'times':
            pts.append(data[key][i])
    pts = np.array(pts)
    time = data['times'][i]

    return time, np.array(pts)


def get_average_positions(data, keys, range):
    xavg = []
    yavg = []
    for key in keys:
        if key != 'times':
            dt = np.array(data[key])
            dt = dt[range[0]:range[1]]

            xavg.append(np.mean(dt[:, 0]))
            yavg.append(np.mean(dt[:, 1]))

    return xavg, yavg


def plot_on_frame(frame, time, pts, avg, plotting_mag=1, save_name='frame.png'):
    num_gyros = len(pts)

    x_for_t = pts[:, 0] - avg[:, 0]
    y_for_t = pts[:, 1] - avg[:, 1]

    print 'pts', pts[1]
    print 'avg', avg[1]

    patch = []
    for j in xrange(num_gyros):
        circ = Circle((avg[j, 0], avg[j, 1]),
                      radius=sqrt((plotting_mag * x_for_t[j]) ** 2 + (plotting_mag * y_for_t[j]) ** 2))
        patch.append(circ)
    colors = (np.arctan2(y_for_t, x_for_t)) % (2 * pi)

    x_for_t = xs + plotting_mag * x_for_t
    y_for_t = ys + plotting_mag * y_for_t

    p = PatchCollection(patch, cmap='isolum_rainbow', alpha=0.7)
    p.set_array(P.array(colors))
    p.set_clim([0, 2 * pi])

    # axes constructor axes([left, bottom, width, height])
    fig = plt.figure(figsize=(5, 5))
    p_ax = P.axes([0.0, 0.0, 1., 1.])
    plt.imshow(frame, cmap=cm.Greys_r)
    p_ax.axes.get_xaxis().set_ticks([])
    p_ax.axes.get_yaxis().set_ticks([])
    p_ax.add_collection(p)
    abc2 = p_ax.scatter(x_for_t, y_for_t, c='w', alpha=1)
    bbox_props = dict(boxstyle="round", fc="k", ec="0.2", alpha=0)
    p_ax.text(25, 45, '$t = %0.1f s$' % time, ha="left", va="baseline", size=32, bbox=bbox_props, color='w',
              family='sans-serif')
    plt.xlim(0, 580)
    plt.ylim(580, 0)
    plt.savefig(save_name)
    p.remove()
    abc2.remove()
    plt.close()


def adjust_frame(current_frame, min_value, max_value):
    current_frame = np.clip(current_frame, min_value, max_value) - min_value
    current_frame = current_frame / (max_value - min_value)

    return current_frame


if __name__ == '__main__':
    video_path = '/Volumes/labshared2/Lisa/2017_02_20_different_lighting/7p_6A_1.cine'  # '#'/Users/lisa/Dropbox/Research/2017_02_17_data/untracked2/2p00hz_0p0amps_2.cine'  # '/Users/lisa/Dropbox/Research/2017_02_17_data/untracked2/2p00hz_0p0_5p0_ramp.cine'
    root_dir = '/Volumes/labshared2/Lisa/2017_02_20_different_lighting/tracked/7p_6A_1_2/'  # '/Users/lisa/Dropbox/Research/2017_02_17_data/tracked/2p00hz_0p0_5p0_ramp_2/'
    save_directory = os.path.join(root_dir, 'movie_frames')

    mod_list = [32, 16, 8, 4, 2, 1]

    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    try:
        c = cine.Cine(video_path)
    except:
        c = []

    path = os.path.join(root_dir, 'com_data.hdf5')
    data = h5py.File(path, 'r')
    keys = data.keys()
    length = len(data['times'])
    neighborhood = 75

    for j in xrange(len(mod_list)):
        if j == 0:
            prev_mod_list = []
        else:
            prev_mod_list = mod_list[:j]

        avg_x = []
        avg_y = []

        for i in xrange(length):
            prev_mods = i % np.array(prev_mod_list)
            prev_mods = list(prev_mods)
            frame_name = os.path.join(save_directory, '%05d.png' % i)
            if i % mod_list[j] == 0 and not (os.path.isfile(frame_name)):

                try:

                    if i < neighborhood:

                        lb = 0
                    else:
                        lb = i - neighborhood

                    if length - 5 < i + neighborhood:
                        ub = length - 5

                    else:
                        ub = i + neighborhood

                    xs, ys = get_average_positions(data, keys, range=[lb, ub])

                    avg_x.append(xs)
                    avg_y.append(ys)
                    time, pts = get_frame_points_and_time(data, keys, i)

                    print len(c)
                    if len(c) > 0:
                        frame = c[i].astype('f')
                    else:

                        frame = np.zeros((600, 600))

                    plot_on_frame(frame, time, pts, np.array([xs, ys]).T, save_name=frame_name, plotting_mag=5)
                except:
                    test = 1

    avg_x = np.array(avg_x)
    avg_y = np.array(avg_y)

    # ffmpeg -r 60 -start_number 0 -i %05d.png -f mp4 h264 -pix_fmt yuv420p video_s.mp4
