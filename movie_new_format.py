import cine
import h5py
import isolum_rainbow
import matplotlib.cm as cm
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import pylab as P
import sys
import time
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Circle, Wedge, Polygon, Rectangle
from subprocess import call


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


def plot_on_frame(frame, time, pts, avg, plotting_mag=1, save_name='frame.png', color_by='phase',
                  cmap='isolum_rainbow'):
    frame = adjust_frame(frame, 0.1, 0.7)

    num_gyros = len(pts)

    x_for_t = pts[:, 0] - avg[:, 0]
    y_for_t = pts[:, 1] - avg[:, 1]
    patch = []
    bol = []
    colors = []
    for j in xrange(num_gyros):

        circ = Circle((avg[j, 0], avg[j, 1]),
                      radius=np.sqrt((plotting_mag * x_for_t[j]) ** 2 + (plotting_mag * y_for_t[j]) ** 2))

        if color_by == 'amplitude':
            colors.append(np.sqrt((plotting_mag * x_for_t[j]) ** 2 + (plotting_mag * y_for_t[j]) ** 2))

        bol.append(True)

    if color_by == 'phase':
        colors = (np.arctan2(y_for_t, x_for_t)) % (2 * np.pi)
        p = PatchCollection(patch, cmap=cmap, alpha=0.6)
        p.set_array(np.array(colors))
        p.set_clim([0, 2 * np.pi])
    else:
        colors = np.array(colors)
        # colors = np.sqrt((plotting_mag* x_for_t) ** 2 + (plotting_mag*y_for_t) ** 2)
        p = PatchCollection(patch, cmap=cmap, alpha=0.6)
        p.set_array(np.array(colors))
        p.set_clim([0, 30])

    x_for_t = avg[:, 0] + plotting_mag * x_for_t
    y_for_t = avg[:, 1] + plotting_mag * y_for_t

    #


    bol = np.array(bol, dtype=bool)
    # axes constructor axes([left, bottom, width, height])
    fig = plt.figure(figsize=(5, 5))
    p_ax = P.axes([0.0, 0.0, 1., 1.])
    plt.imshow(frame, cmap=cm.Greys_r)
    p_ax.axes.get_xaxis().set_ticks([])
    p_ax.axes.get_yaxis().set_ticks([])
    p_ax.add_collection(p)
    abc2 = p_ax.scatter(x_for_t[bol], y_for_t[bol], c='w', alpha=1)
    bbox_props = dict(boxstyle="round", fc="k", ec="0.2", alpha=0)
    p_ax.text(25, 45, '$t = %0.1f s$' % time, ha="left", va="baseline", size=32, bbox=bbox_props, color='w',
              family='sans-serif')
    plt.xlim(0, 580)
    plt.ylim(580, 0)
    plt.savefig(save_name)

    abc2.remove()

    plt.close()


def adjust_frame(current_frame, min_value, max_value):
    max = np.max(current_frame.astype('float').flatten())
    min_value = min_value * max
    max_value = max_value * max

    current_frame = np.clip(current_frame, min_value, max_value) - min_value
    current_frame = current_frame / (max_value - min_value)

    return current_frame


def make_frames(root_dir, video_path, fn_for_movie='video.mp4', color_by='phase', cmap='isolum_rainbow'):
    save_directory = os.path.join(root_dir, 'movie_frames_' + color_by + '_' + cmap)

    mod_list = [16, 8, 4, 2, 1]

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
        avg_x = []
        avg_y = []

        for i in xrange(length):
            frame_name = os.path.join(save_directory, '%05d.png' % i)
            if i % mod_list[j] == 0 and not (os.path.isfile(frame_name)):

                if True:

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

                    if len(c) > 0:
                        frame = c[i].astype('f')
                    else:

                        frame = np.zeros((600, 600))

                    plot_on_frame(frame, time, pts, np.array([xs, ys]).T, save_name=frame_name, plotting_mag=5,
                                  color_by=color_by, cmap=cmap)
            if mod_list[j] == 1 and i >= length - 2:
                call(["ffmpeg", "-r", "50", "-start_number", str(0), "-i", save_directory + "/%05d.png", "-f", "mp4",
                      save_directory + "h264", "-pix_fmt", "yuv420p", os.path.join(save_directory, fn_for_movie)])
                # call(
                #   ["ffmpeg", "-r", "50", "-i", "video.mp4", output_dir + final_fn_for_movie + '.mp4'])


# def draw_progress(ax, data):


if __name__ == '__main__':
    video_path = '/Volumes/labshared2/Lisa/2017_02_21/7p_0p0A_5p5A_1.cine'  # '#'/Users/lisa/Dropbox/Research/2017_02_17_data/untracked2/2p00hz_0p0amps_2.cine'  # '/Users/lisa/Dropbox/Research/2017_02_17_data/untracked2/2p00hz_0p0_5p0_ramp.cine'
    root_dir = '/Volumes/labshared2/Lisa/2017_02_21/tracked/7p_0p0A_5p5A_1/'  # '/Users/lisa/Dropbox/Research/2017_02_17_data/tracked/2p00hz_0p0_5p0_ramp_2/'


    # ffmpeg -r 120 -start_number 0 -i %05d.png -f mp4 h264 -pix_fmt yuv420p video_s.mp4
