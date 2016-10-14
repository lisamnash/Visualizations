from functions import *
from plot_on_movie_functions import *
import os
import time
from subprocess import call
import cine
import sys


if __name__ == '__main__':

        
    mod_list = [32, 16, 8, 4, 2, 1]
    len_mod_list = len(mod_list)
    
    dirs = [ '/Volumes/GetIt/saved_stuff/2016_09_08_double_magnet_videos/1p55_1p5_1p40_double_boundary_Cam_7326_Cine3']

    
    video_paths = ['/Volumes/GetIt/saved_stuff/2016_09_08_double_magnet_videos/untracked/1p55_1p5_1p40_double_boundary_Cam_7326_Cine3.cine']

    
    for i in range(len(video_paths)):
        times_ap = []

        fn_for_movie = 'video'

        f0 = 0 #first frame for movie
        

    
        video_path = video_paths[i]
        final_fn_for_movie = video_path.split('/')[-1]
     
        final_fn_for_movie = final_fn_for_movie.split('.')[0]
       
        base_dir = dirs[i]
      
      
        plot_stuff, times = compile_x_y_traces(base_dir)
        
        first_time = min(times)
    
        try :
            c = cine.Cine(video_path)
            
            times_movie = []
            len_movie = len(c)
            for ii in range(len_movie):
                times_movie.append(c.get_time(ii))
            
            #here we find the first frame that was tracked
            ftf =  where(abs(times_movie-first_time) < 10**-6)[0][0]
            ftf_fmf_diff =  f0 -ftf
            if f0 < ftf:
                print 'First movie frame (%04d) is before first tracked frame(%04d).  Exiting' %(f0, ftf)
                sys.exit()
         


            output_dir = base_dir + '/plot_on_movie_f_4smooth_7times/'

            if not(os.path.isfile(output_dir + 'done.pickle')):
                for j in range(len_mod_list):
                    if j == 0:
                        prev_mod_list = []
                    else :
                        prev_mod_list = mod_list[:j]
                    len_times = len(times) - ftf_fmf_diff
                    for k in range(len_times):
                        prev_mods = k%array(prev_mod_list)
                        prev_mods = list(prev_mods)
                        if k%mod_list[j] == 0 and not(os.path.isfile(output_dir + '%04d.png' %k)):
                            t = plot_on_movie_circle_single_frame(c, f0 + k, times[ftf_fmf_diff+k], [1,0.75], output_dir, k, 7, ftf, plot_stuff)
                            times_ap.append([t,k])
                    if mod_list[j] == 1 and k == range(len_times)[-1]:
                        dump_pickled_data(output_dir, 'done' , 1)
                        times_ap_arr = array(times_ap)
                      
                        call(['cd', output_dir])
                        call(['pwd'])
                
                        
                        call(["ffmpeg", "-start_number", str(0), "-i" , output_dir+"%04d.png", "-f", "mp4", output_dir+ "h264", "-pix_fmt" ,"yuv420p", output_dir+fn_for_movie + ".mp4"])
                        call(["ffmpeg", "-r", "60", "-i" , "video.mp4",  output_dir +  final_fn_for_movie + '.mp4' ])
                        print output_dir+fn_for_movie + ".mp4"
        except RuntimeError:
            sys.exit()
                