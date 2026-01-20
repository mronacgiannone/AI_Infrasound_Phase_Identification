# Deep Clustering for Infrasound Phase Identification
#
# McAAP
#
# Miro Ronac Giannone (mronacgiannone@smu.edu)
#-----------------------------------------------------------------------------------------------------------------#
# Import pacakages
import glob, datetime, warnings, sys, fnmatch, pywt, os, keras
sys.path.append('/users/mronacgiannone/Documents/Cardinal')
import cardinal
#-----------------------------------------------------------------------------------------------------------------#
# Import packages as
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------------------------------------------------#
# Import functions from packages
from obspy import *
from obspy.core import *
from scipy import signal
from PIL import Image as Image_PIL
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, adjusted_rand_score, adjusted_mutual_info_score, confusion_matrix, normalized_mutual_info_score, silhouette_score
# from pyproj import Geod; g = Geod(ellps='sphere') # could not import this properly so g.inv() was also removed in the beamforming code below
#-----------------------------------------------------------------------------------------------------------------#
# ML Packages 
import tensorflow as tf
from keras.utils import *
from tensorflow.keras import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
#-----------------------------------------------------------------------------------------------------------------#
# Ignore non-critical warnings
warnings.filterwarnings("ignore")

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def read_data(day, data_dir, loc_dir, instrument_sensitivity=224000, Pa_conversion=0.0035012, MCA=False, MCB=False, EOCR=False, EOCL=False, WCT=False):
    
    # Download data and remove instrument response
    df = pd.read_table(loc_dir, header=None, sep=r'\s+', names=['stn', 'lat', 'lon']) # Coordinates
    st = Stream()
    if MCA == True:
        for filepath in glob.iglob(data_dir[0]+'*'+str(day)):
            tr = read(filepath)
            tr = tr.merge()
            tr[0].data = tr[0].data/56000 # Removing instrument response based on instrument sensitivity in XML files
            st.append(tr[0])
        for filepath in glob.iglob(data_dir[1]+'*'+str(day)):
            tr = read(filepath)
            if len(tr) == 0: # some of the outer elements stopped recording
                continue
            tr = tr.merge()
            tr[0].data = tr[0].data/instrument_sensitivity # Removing instrument response
            st.append(tr[0])
    elif MCB == True:
        for filepath in glob.iglob(data_dir+'MCB*/BDF*/'+'*'+str(day)):
            tr = read(filepath)
            if len(tr) == 0:
                continue
            tr = tr.merge()
            tr[0].data = tr[0].data/instrument_sensitivity # Removing instrument response
            st.append(tr[0])
    elif EOCR == True:
        for filepath in glob.iglob(data_dir+'EOC*/BDF.D/SM.EOC*'+day):
            tr = read(filepath)
            tr = tr.merge()
            tr[0].data = tr[0].data/224000 # Removing instrument response
            st.append(tr[0])
    elif EOCL == True:
        with open(data_dir+'MCA_Detections/Signal_Times_CWT/UTC_Starttimes/'+str(day)+'.txt', 'r') as f: # reference time (UTC)
            ref_time = UTCDateTime(f.read())
        for filepath in glob.iglob(data_dir+'EOC_line_mseed/'+str(ref_time).split('T')[0]+'*L*'):
            tr = read(filepath)
            tr = tr.merge()
            tr[0].data = tr[0].data * Pa_conversion # Converting to Pa
            st.append(tr[0])
    elif WCT == True:
        for filepath in glob.iglob(data_dir+'WCT*/BDF.D/SM.WCT*'+day):
            tr = read(filepath)
            tr = tr.merge()
            if (len(tr) == 0) or (tr[0].stats.station == 'WCT06'): # WCT06 too noisy to detect signals
                continue
            tr[0].data = tr[0].data/instrument_sensitivity # Removing instrument response
            st.append(tr[0])
    #-----------------------------------------------------------------------------------------------------------------------#
    # Append location info
    for tr in st:
        index = np.where(df['stn'] == tr.id)[0]
        sacAttrib = AttribDict({"stla": df['lat'][index],
                                "stlo": df['lon'][index]})
        tr.stats.sac = sacAttrib
    for tr in st:
        lat = (df[df['stn'] == tr.id]['lat']).values[0]
        lon = (df[df['stn'] == tr.id]['lon']).values[0]
        tr.stats.sac.stla = lat
        tr.stats.sac.stlo = lon
        tr.stats.sac.stel = 0
    #-----------------------------------------------------------------------------------------------------------------------#
    return st

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def YMD_to_JD(date_format, date):
    '''---------------------------------------------------------------------------
    Converts calendar date from YYYY/MM/DD format to julian day and year.
    ---------------------------------------------------------------------------'''
    dt = datetime.datetime.strptime(date, date_format)
    tt = dt.timetuple()
    julian_day = tt.tm_yday; year = dt.year
    #-----------------------------------------------------------------------------------------------------------------------#
    return julian_day, year

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def get_signal_times_cwt(day_idx, dets_julian_days, freqmin=1, freqmax=10, scales=[0.1,0.2,0.3,0.4,0.5], min_snr=10, min_distance_secs=15, MCA=False, MCB=False, WCT=False, vel=None,
                         source_lon=-95.902050, source_lat=34.802819, x_lim=None, y_lim=[-1,1.05], trim_from_start=None, trim_from_end=None, find_peaks=False, save=False, normalize=True,
                         legend_loc='upper right', legend_size=10, figsize=(9,6), remove_stations=None):
    
    # Read trimmed data and filter
    if MCA == True:
        directory = '/Volumes/Extreme SSD/McAAP/Winter_2021/MCA_Detections/Data/MCA_'+str(dets_julian_days[day_idx])+'.mseed'; st = read(directory)
    elif MCB == True:
        directory = '/Volumes/Extreme SSD/McAAP/Summer_2021/MCB_Detections/Data/MCB_'+str(dets_julian_days[day_idx])+'.mseed'; st = read(directory)
    elif WCT == True:
        # Need to append location info here for beamforming
        directory = '/Volumes/Extreme SSD/McAAP/Summer_2021/WCT_Detections/Data/WCT_'+str(dets_julian_days[day_idx])+'.mseed'; st = read(directory)
        df = pd.read_table('/Volumes/Extreme SSD/McAAP/Summer_2021/WCT_BDF_Locations.txt', header=None, sep=r'\s+', names=['stn', 'lat', 'lon']) # Coordinates
        for tr in st:
            index = np.where(df['stn'] == tr.id)[0]
            sacAttrib = AttribDict({"stla": df['lat'][index],
                                    "stlo": df['lon'][index]})
            tr.stats.sac = sacAttrib
        for tr in st:
            lat = (df[df['stn'] == tr.id]['lat']).values[0]
            lon = (df[df['stn'] == tr.id]['lon']).values[0]
            tr.stats.sac.stla = lat
            tr.stats.sac.stlo = lon
            tr.stats.sac.stel = 0
        if remove_stations is not None:
            for tr in st:
                for remove_stn_idx in range(len(remove_stations)):
                    if tr.stats.station == remove_stations[remove_stn_idx]:
                        st.remove(tr)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Further trimming stream
    if trim_from_start is not None:
        start_date = str(st[0].stats.starttime); end_date = str(st[0].stats.endtime)
        dt_start = UTCDateTime(start_date); dt_end = UTCDateTime(end_date)
        st = st.trim(dt_start+trim_from_start, dt_end)
    if trim_from_end is not None:
        start_date = str(st[0].stats.starttime); end_date = str(st[0].stats.endtime)
        dt_start = UTCDateTime(start_date); dt_end = UTCDateTime(end_date)
        st = st.trim(dt_start, dt_end-trim_from_end)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Filter
    st_filt = st.copy()
    st_filt.taper(type='cosine', max_percentage=0.05, max_length=60)
    try:
        st_filt.filter('bandpass', freqmin=freqmin, freqmax=freqmax)
    except:
        st_filt = st_filt.split()
        st_filt.filter('bandpass', freqmin=freqmin, freqmax=freqmax)
        st_filt.merge()
    #-----------------------------------------------------------------------------------------------------------------------#
    # Plot - since signal times are relative make sure to only trim end of stream and not start
    if (MCA == True) or (MCB == True):
        t = np.arange(0, st[0].stats.delta*st[0].stats.npts, st[0].stats.delta)
        if len(st_filt) > 4: st_filt = st_filt.merge()
        t_data, data = cardinal.data_time_window(t, st_filt, 0, st[0].stats.endtime - st[0].stats.starttime)
        t_data, _ = cardinal.fix_lengths(t_data, st_filt[0].data)
        fig = plt.figure(figsize = figsize)
        for i, tr in enumerate(st_filt):
            if i == 0:
                ax_tmp = fig.add_subplot(len(st_filt),1,i+1)
            else:
                ax_tmp = fig.add_subplot(len(st_filt),1,i+1, sharex=ax_tmp, sharey=ax_tmp)
            if normalize == True:
                data_tmp = data[i,:] / np.abs(data[i,:].max()) # Normalizing waveforms
            else:
                data_tmp = data[i,:].copy()
            if MCA == True:
                if tr.stats.channel == 'BDF': label_tmp = 'MCA01'
                elif tr.stats.channel == 'HHZ': label_tmp = 'MCA02'
                elif tr.stats.channel == 'HHE': label_tmp = 'MCA03'
                elif tr.stats.channel == 'HHN': label_tmp = 'MCA04'
            elif MCB == True:
                label_tmp = tr.stats.station
            plt.plot(t_data, data_tmp, label=label_tmp)
            #-----------------------------------------------------------------------------------------------------------------------#
            if find_peaks == True:
                # Finding signal peaks
                peaks = signal.find_peaks_cwt(data_tmp, widths=scales, min_snr=min_snr)
                #-----------------------------------------------------------------------------------------------------------------------#
                # Filter peaks to enforce minimum distance
                filtered_peaks = []
                last_peak = -np.inf  # Initialize to a large negative value
                min_distance = min_distance_secs*tr.stats.sampling_rate # convert to seconds
                for peak in peaks:
                    if peak - last_peak > min_distance:
                        filtered_peaks.append(peak)
                        last_peak = peak
                filtered_peaks = np.array(filtered_peaks)
                plt.plot(filtered_peaks/tr.stats.sampling_rate, data_tmp[filtered_peaks], 'x')
            #-----------------------------------------------------------------------------------------------------------------------#
            # Plotting params
            if i == len(st_filt)-1:
                ax_tmp.set_xlabel('Time (s) after ' + str(tr.stats.starttime).split('.')[0].replace('T', ' '))
                if normalize == True:
                    ax_tmp.set_ylabel('Normalized\nPressure')
                else:
                    ax_tmp.set_ylabel('Pressure [Pa]')
            else:
                ax_tmp.tick_params(labelbottom=False)
            plt.ylim(y_lim)
            if x_lim is not None:
                plt.xlim(x_lim)
            else:
                plt.xlim([0, st_filt[0].stats.endtime - st_filt[0].stats.starttime])
            plt.legend(loc=legend_loc, prop={ "size": legend_size})
            #-----------------------------------------------------------------------------------------------------------------------#
            # Saving source array peaks
            if MCA == True:
                outdir = '/Volumes/Extreme SSD/McAAP/Winter_2021/MCA_Detections/Signal_Times_CWT/'
                outfigdir = outdir + 'Plots/MCA_'
            elif MCB == True:
                outdir = '/Volumes/Extreme SSD/McAAP/Summer_2021/MCB_Detections/Signal_Times_CWT/'
                outfigdir = outdir + 'Plots/MCB_'
            if (save == True) & (find_peaks==True):
                with open(outdir+tr.stats.station+'_'+tr.stats.channel+'/'+str(dets_julian_days[day_idx])+'.npy', 'wb') as f:
                    peaks_in_seconds = np.array([filtered_peaks/tr.stats.sampling_rate]) # save in seconds (otherwise it saves in samples idx)
                    if trim_from_start is not None:
                        np.save(f, peaks_in_seconds+trim_from_start) 
                    else:
                        np.save(f, peaks_in_seconds)
                print('Peaks saved for '+ tr.stats.station + ' - '+ tr.stats.channel)
            plt.suptitle(outfigdir[-4:-1] +' '+str(st_filt[0].stats.starttime).split('T')[0])
            if (save == True) & (find_peaks==True):
                plt.savefig(outfigdir+str(dets_julian_days[day_idx]))
                print('Figure saved')
    #-----------------------------------------------------------------------------------------------------------------------#
    # WCT Best Beam
    elif WCT == True:
        # _, GT_baz, _ = g.inv(source_lon, source_lat, st[0].stats.sac.stlo, st[0].stats.sac.stla) # using midpoint between areas 1 and 2
        GT_baz = 0
        if GT_baz < 0: GT_baz += 360
        t_shifts = cardinal.get_slowness_vector_time_shifts(st, ref_station=st[0].stats.station, baz=GT_baz, vel=vel, units='km')
        _, beam = cardinal.beamform(t_shifts, st_filt, ref_station=st[0].stats.station, plot=True, legend_loc=legend_loc, legend_size=legend_size, normalize_data=normalize, normalize_beam=normalize,
                                         figsize=figsize)
        if find_peaks == True:
            # Finding signal peaks
            peaks = signal.find_peaks_cwt(beam, widths=scales, min_snr=min_snr)
            #-----------------------------------------------------------------------------------------------------------------------#
            # Filter peaks to enforce minimum distance
            filtered_peaks = []
            last_peak = -np.inf  # Initialize to a large negative value
            min_distance = min_distance_secs*st[0].stats.sampling_rate # convert to seconds
            for peak in peaks:
                if peak - last_peak > min_distance:
                    filtered_peaks.append(peak)
                    last_peak = peak
            filtered_peaks = np.array(filtered_peaks)
            plt.plot(filtered_peaks/st[0].stats.sampling_rate, beam[filtered_peaks], 'x', color='red')
        #-----------------------------------------------------------------------------------------------------------------------#
        # Plotting params
        if normalize == True:
            plt.ylabel('Normalized\nPressure')
        else:
            plt.ylabel('Pressure [Pa]')
        if x_lim is not None:
            plt.xlim(x_lim)
        else:
            plt.xlim([0, st_filt[0].stats.endtime - st_filt[0].stats.starttime])
    #-----------------------------------------------------------------------------------------------------------------------#
        # Saving WCT beam peaks
        outdir = '/Volumes/Extreme SSD/McAAP/Summer_2021/WCT_Detections/Beamform_Signal_Times_CWT/'
        outfigdir = outdir + 'Plots/WCT_'
        if (save == True) & (find_peaks==True):
            with open(outdir+'WCT_Beam/'+str(dets_julian_days[day_idx])+'.npy', 'wb') as f:
                peaks_in_seconds = np.array([filtered_peaks/st[0].stats.sampling_rate]) # save in seconds (otherwise it saves in samples idx)
                if trim_from_start is not None:
                    np.save(f, peaks_in_seconds+trim_from_start) 
                else:
                    np.save(f, peaks_in_seconds)
            print('Peaks saved for WCT beam')
        plt.suptitle(outfigdir[-4:-1] +' '+str(st_filt[0].stats.starttime).split('T')[0])
        plt.tight_layout()
        if (save == True) & (find_peaks==True):
            plt.savefig(outfigdir+str(dets_julian_days[day_idx]))
            print('Figure saved')

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def extract_signals_from_dict(row_dict, index):
    
    if isinstance(index, int):
        signal = list(row_dict.items())[index] # returns tuple
    elif isinstance(index, str):
        signal = {key: value for key, value in row_dict.items() if fnmatch.fnmatch(key, index)} # returns dict
    elif isinstance(index, list):
        signal = []
        for sig_idx in range(len(index)):
            signal_tmp = list(row_dict.items())[sig_idx] # returns tuple
            signal.append(signal_tmp)

    return signal

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def separate_dict(array_dict, data_npts=1200):
    
    # Make data and labels separately for each array
    X = np.zeros((len(array_dict),data_npts))
    y = []
    for i, dict_idx in enumerate(array_dict):
        y.append(dict_idx) # saves each string to list
        X[i,:] = array_dict[dict_idx]
    y = np.array(y)
    
    return X, y

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def get_beam_for_xcorr(data_dir, array, julian_day, freqmin=1, freqmax=10, source_lon=-95.902050, source_lat=34.802819, legend_loc='upper left', legend_size=7.5, normalize=False, xlim=None,
                       plot=True, figsize=(9,6)):
    
    # Read data
    directory = data_dir+array+'_Detections/Data/'+array+'_'+str(julian_day)+'.mseed'; st = read(directory)
    if array == 'EOCL':
        df = pd.read_table(data_dir+'/'+array+'_HDF_Locations.txt', header=None, sep=r'\s+', names=['stn', 'lat', 'lon']) # Coordinates
    else:
        df = pd.read_table(data_dir+'/'+array+'_BDF_Locations.txt', header=None, sep=r'\s+', names=['stn', 'lat', 'lon']) # Coordinates
    #-----------------------------------------------------------------------------------------------------------------------#
    # Append location info
    for tr in st:
        index = np.where(df['stn'] == tr.id)[0]
        sacAttrib = AttribDict({"stla": df['lat'][index],
                                "stlo": df['lon'][index]})
        tr.stats.sac = sacAttrib
    for tr in st:
        lat = (df[df['stn'] == tr.id]['lat']).values[0]
        lon = (df[df['stn'] == tr.id]['lon']).values[0]
        tr.stats.sac.stla = lat
        tr.stats.sac.stlo = lon
        tr.stats.sac.stel = 0
    #-----------------------------------------------------------------------------------------------------------------------#
    # Relabel stations
    if array == 'MCA':
        for tr in st:
            if tr.stats.channel == 'BDF': tr.stats.station = 'MCA01'; tr.stats.location = ''
            elif tr.stats.channel == 'HHZ': tr.stats.station = 'MCA02'; tr.stats.channel = 'BDF'
            elif tr.stats.channel == 'HHE': tr.stats.station = 'MCA03'; tr.stats.channel = 'BDF'
            elif tr.stats.channel == 'HHN': tr.stats.station = 'MCA04'; tr.stats.channel = 'BDF'
    #-----------------------------------------------------------------------------------------------------------------------#
    # Filter data
    st_filt = st.merge().copy()
    st_filt.taper(type='cosine', max_percentage=0.05, max_length=60)
    try:
        st_filt.filter('bandpass', freqmin=freqmin, freqmax=freqmax)
    except:
        st_filt = st_filt.split()
        st_filt.filter('bandpass', freqmin=freqmin, freqmax=freqmax)
        st_filt.merge()
    #-----------------------------------------------------------------------------------------------------------------------#
    # Beamform
    # _, GT_baz, _ = g.inv(source_lon, source_lat, st[0].stats.sac.stlo, st[0].stats.sac.stla) # using midpoint between areas 1 and 2
    GT_baz = 0
    if GT_baz < 0: GT_baz += 360
    t_shifts = cardinal.get_slowness_vector_time_shifts(st, ref_station=st[0].stats.station, baz=GT_baz, tr_vel=0.35, units='km')
    t_beam, beam = cardinal.beamform(t_shifts, st_filt, ref_station=st[0].stats.station, plot=plot, legend_loc=legend_loc, legend_size=legend_size, normalize_data=normalize, 
                                        normalize_beam=normalize, figsize=figsize)
    if xlim is not None:
        plt.xlim(xlim)
    if plot == True:
        plt.ylabel('Pressure (Pa)')
    st_beam = cardinal.add_beam_to_stream(st_filt, beam, ref_station=st[0].stats.station)
    #-----------------------------------------------------------------------------------------------------------------------#
    return t_beam, st_beam

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def get_signals_from_beam(t_beam, st_beam, data_dir, filepath, time_to_start=15, time_to_end=15):
    
    # Load onset times
    signal_onset_time_filepath = data_dir+filepath
    if st_beam[0].stats.station[:3] == 'EOC':
        times = pd.read_csv(signal_onset_time_filepath)
    else:
        with open(signal_onset_time_filepath, 'rb') as f: # loading signal onset times
            times = np.load(f)[0]
    #-----------------------------------------------------------------------------------------------------------------------#
    # Extract data from each trace using 30 second windows (+/- 15 seconds from signal onset time)
    data = []
    st_beam_tmp = Stream().append(st_beam[-1]) # using only beamformed data
    for sig_idx in range(len(times)):
        if st_beam[0].stats.station[:3] == 'EOC':
            sig_time_tmp = (times['end_time'][sig_idx] + times['start_time'][sig_idx]) / 2 # use midpoint between Cardinal families' start and end times
            t_data, data_tmp = cardinal.data_time_window(t_beam, st_beam_tmp, t_start=sig_time_tmp-time_to_start, t_end=sig_time_tmp+time_to_end)
        else:
            t_data, data_tmp = cardinal.data_time_window(t_beam, st_beam_tmp, t_start=times[sig_idx]-time_to_start, t_end=times[sig_idx]+time_to_end)
        # Desired length
        target_length = int(st_beam[0].stats.sampling_rate) * (time_to_start+time_to_end) # sampling rate and time window length
        if data_tmp.shape[1] != target_length:
            # Calculate padding required on each side
            pad_width = target_length - data_tmp.shape[1]
            # Pad the array on both sides to reach target length
            data_tmp = np.pad(data_tmp, (0, pad_width), mode='constant')
        data.append(data_tmp[0,:])
    data = np.array(data)
    #-----------------------------------------------------------------------------------------------------------------------#
    return t_data, data

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def norm_xcorr_temporal(t_data, data1, data2):

    a = data1; b = data2
    a = (a - np.mean(a)) / (np.std(a) * len(a))
    b = (b - np.mean(b)) / (np.std(b))
    x_corr = np.correlate(a, b, 'full')
    lags = np.hstack((np.flipud(-t_data)[0:len(t_data)-1],t_data))
    return lags, x_corr

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def xcorr_matrix_temporal(t_data, data):

    xcorr_coef = []
    for i in range(0,len(data)):
        xcorr_coef_i = []
        for m in range(len(data)):
            lags, xcorr = norm_xcorr_temporal(t_data, data[i,:], data[m,:])
            xcorr_coef_i.append(max(xcorr))
        xcorr_coef.append(xcorr_coef_i)
    xcorr_coef = np.array(xcorr_coef)
    xcorr_coef_mean = xcorr_coef.mean(1)

    ref_signal = np.where(xcorr_coef_mean == np.max(xcorr_coef_mean)) # Choosing reference signal based on highest average correlation 

    xcorr_lag_times = []
    for i in range(0,len(data)):
        xcorr_lag_times_i = []
        for m in range(len(data)):
            lags, xcorr = norm_xcorr_temporal(t_data, data[i,:], data[m,:])
            max_idx = np.where((xcorr == xcorr.max()))[0]
            lag_times = lags[max_idx]; ymax = xcorr[max_idx]
            xcorr_lag_times_i.append(lag_times)
        xcorr_lag_times.append(xcorr_lag_times_i)
    xcorr_lag_times = np.array(xcorr_lag_times)
    lag_tshifts = xcorr_lag_times[:, :, 0]

    return xcorr_coef, lag_tshifts, ref_signal

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def plot_LagShifts(st, t_data, data, lag_tshifts, ref_signal, mdir, datadir, outfigdir, name_of_array):

    fig = plt.figure(figsize=(12,4))

    ax1 = fig.add_subplot(1,1,1)
    
    if not os.path.isdir(outfigdir):
        os.mkdir(outfigdir)
    
    # Removing previous files -- Careful with this
    pngfiles=glob.glob(datadir+'/*.png')
    for f in pngfiles:
        os.remove(f)
    
    n = len(data)
    colors = plt.cm.hot_r(np.linspace(0,1,n))
    colors2 = plt.cm.Greys_r(np.linspace(0,1,n))
    x_upper = len(t_data)/st[0].stats.sampling_rate

    for i in range(n):
        plt.plot(t_data + lag_tshifts[ref_signal[0],i], data[i,:]/np.max(data[i,:]), color=colors[i])
        plt.title('Waveform Morphology')
        plt.xlim([0, x_upper])
        plt.ylim([-1.05, 1.05])
        ax1 = plt.gca()
        ax1.set_facecolor(colors2[i])
        plt.xlabel('Time (s)')
        plt.ylabel('Normalized Amplitude')
        outf=outfigdir+str(i/10)+'.png'
        plt.savefig(outf)
    
    frames=[]
    imgs=sorted(glob.glob(datadir+'/*.png'))

    for i in imgs:
        new_frame=Image_PIL.open(i)
        frames.append(new_frame)
    
    frames[0].save(datadir + name_of_array + '.gif',format='GIF',append_images=frames[1:],save_all=True,duration=700,loop=0)

    stream = datadir + name_of_array + '.gif'

    return stream

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def plot_XcorrMatrix(st, xcorr_coef, name_of_array = None, pit_number = None, startTimes = None, vmin=0.3, vmax=1, cmap='hot_r', figsize=(30,10)):

    windows = []

    if pit_number is not None:
        for i in range(len(np.array(pit_number))):
            windows.append(pit_number[i])
        ticks = np.arange(0,len(np.array(pit_number)),1)

    if startTimes is not None:
        for i in range(len(np.array(startTimes))):
            windows.append(int(np.round(startTimes[i],-1)))
        ticks = np.arange(0,len(np.array(startTimes)),1)

    f1, ax = plt.subplots(1, 1, figsize=figsize)

    plt.pcolormesh(xcorr_coef, vmin=vmin, vmax=vmax, cmap=cmap)
    colorbar = plt.colorbar(ax=ax)
    colorbar.set_label('Normalized Correlation Coefficient',fontsize=12)

    plt.xticks(ticks,windows,ha='left')
    plt.yticks(ticks,windows,va='baseline')
    plt.xlabel('Time (s) after ' + str(st[0].stats.starttime).split('.')[0].replace('T', ' '))
    plt.ylabel('Time (s) after ' + str(st[0].stats.starttime).split('.')[0].replace('T', ' '))
    if name_of_array is not None:
        plt.title(name_of_array + ' Cross-Correlation')

    plt.show()

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def scales_for_freq_grid(wavelet, fs, f_lo=1.0, f_hi=10.0, n_freq=64):
    """
    Linear frequency grid in Hz -> CWT scales.
    ~0.14 Hz spacing when n_freq=64 across 1–10 Hz.
    """
    freqs = np.linspace(f_lo, f_hi, n_freq)   # Hz
    fc = pywt.central_frequency(wavelet)      # cycles/sample
    dt = 1.0 / fs
    scales = fc / (freqs * dt)                # scale = fc / (f * dt)
    return scales, freqs

def preprocess_signals(X_data, X_labels, sampling_rate=40, wavelet='cmor0.25-1.75', f_lim=[1,10], n_freq=64):
    
    fs = float(sampling_rate)
    # ---- NEW: build scales from a linear frequency grid ----
    scales, freq_grid = scales_for_freq_grid(wavelet, fs, f_lo=f_lim[0], f_hi=f_lim[1], n_freq=n_freq)
    X_wavf = []; X_cwt = []; X_envelope = []; X_phase = []; X_label = []
    for sig_idx in range(X_data.shape[0]):
        # Extract signal
        signal_tmp = X_data[sig_idx,:]
        signal_tmp = signal_tmp / (np.max(np.abs(signal_tmp)) + 1e-12) # normalize signal
        #-----------------------------------------------------------------------------------------------------------------------#
        # Convert to scalogram - scales defined outside loop
        coefs, frequencies = pywt.cwt(signal_tmp, scales, wavelet, sampling_period=1/fs)
        scalogram = (np.abs(coefs)) ** 2 # compute scalogram power spectrum (squared magnitude)
        scalogram = np.log10(scalogram + 1e-10)  # Log normalization (small constant added to avoid log(0))
        instant_phase = np.angle(coefs) # compute the instantaneous phase in radians using cwt output
        #-----------------------------------------------------------------------------------------------------------------------#
        # Keep only freqs of interest specified by scalogram range
        mask = (frequencies >= f_lim[0]) & (frequencies <= f_lim[1])
        idx = np.where(mask)[0]
        frequencies = frequencies[idx]
        scalogram = scalogram[idx, :]
        instant_phase = instant_phase[idx, :]
        #-----------------------------------------------------------------------------------------------------------------------#
        # Compute Hilbert envelope and instantaneous phase
        analytic_signal = signal.hilbert(signal_tmp) # compute the Hilbert transform (analytic signal)
        signal_envelope = np.abs(analytic_signal) # compute the envelope (magnitude of the analytic signal)
        #-----------------------------------------------------------------------------------------------------------------------#
        # Appending waveform and scalogram results
        X_wavf.append(np.nan_to_num(signal_tmp)) # normalized waveform
        X_cwt.append(np.nan_to_num(scalogram))
        X_envelope.append(np.nan_to_num(signal_envelope))
        X_phase.append(np.nan_to_num(instant_phase))
        X_label.append(X_labels[sig_idx])
    #-----------------------------------------------------------------------------------------------------------------------#
    # Printing array sizes
    X_wavf = np.array(X_wavf); X_cwt = np.array(X_cwt) 
    X_envelope = np.array(X_envelope); X_phase = np.array(X_phase); X_label = np.array(X_label)
    print('Waveform dataset shape is: '+str(X_wavf.shape))
    print('Scalogram dataset shape is: '+str(X_cwt.shape))
    print('Envelope dataset shape is: '+str(X_envelope.shape))
    print('Phase dataset shape is: '+str(X_phase.shape))

    #-----------------------------------------------------------------------------------------------------------------------#
    # Return arrays
    return X_wavf, X_envelope, X_phase, X_cwt, X_label

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def augment_data(input_feature_vector, max_t_shift=5, noise_percentage=5, sampling_rate=40):
    
    # Convert input to tensorflow tensor
    input_feature_vector = tf.convert_to_tensor(input_feature_vector, dtype=tf.float32)
    #-----------------------------------------------------------------------------------------------------------------------#
    # If the tensor has rank 2, expand it to rank 3 by adding a batch dimension
    if len(input_feature_vector.shape) == 2:
        input_feature_vector = tf.expand_dims(input_feature_vector, axis=-1)  # Adds batch dimension at the end
    #-----------------------------------------------------------------------------------------------------------------------#
    # Add random noise (mean = 0, std = absolute value of the variance of the signal)
    variance = tf.math.reduce_variance(input_feature_vector)
    abs_variance = tf.abs(variance)
    noise_percentage /= 100
    scaled_var = abs_variance * noise_percentage # if you want the scaled std then do tf.sqrt(abs_variance)
    noise = tf.random.normal(mean=0, stddev=scaled_var, shape=tf.shape(input_feature_vector), dtype=tf.float32)
    input_feature_vector += noise
    #-----------------------------------------------------------------------------------------------------------------------#
    # Time shifting signal using random number from -5 to 5 seconds
    t_shift = tf.random.uniform(shape=[], minval=-max_t_shift, maxval=max_t_shift, dtype=tf.float32)
    input_shifted = tf.roll(input_feature_vector, int(t_shift*sampling_rate), axis=1)
    
    return input_shifted

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def trimodal_augment_data(x1, x2, x3, x4, max_t_shift=5, noise_percentage=5.0, sampling_rate=40, trim_sec=5):
    # Cast to float32
    x1 = tf.cast(x1, tf.float32)
    x2 = tf.cast(x2, tf.float32)
    x3 = tf.cast(x3, tf.float32)
    x4 = tf.cast(x4, tf.float32)

    # Ensure 3D with channel-last = 1 if needed (time is axis=1)
    if len(x1.shape) == 2: x1 = tf.expand_dims(x1, axis=-1)
    if len(x2.shape) == 2: x2 = tf.expand_dims(x2, axis=-1)
    if len(x3.shape) == 2: x3 = tf.expand_dims(x3, axis=-1)
    if len(x4.shape) == 2: x4 = tf.expand_dims(x4, axis=-1)

    # Add Gaussian noise (no clipping; you’re z-scored)
    def add_noise(x, pct):
        # per-sample std over freq/time dims -> shape broadcastable back to x
        axes = [1, 2] if x.shape.rank >= 3 else [0]
        std = tf.math.reduce_std(x, axis=axes, keepdims=True)
        noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=std * (pct / 100.0))
        return x + noise

    if noise_percentage and noise_percentage > 0:
        x1 = add_noise(x1, noise_percentage)
        x2 = add_noise(x2, noise_percentage)
        x3 = add_noise(x3, noise_percentage)
        x4 = add_noise(x4, noise_percentage)

    # Shared time shift (seconds → samples)
    if max_t_shift and max_t_shift > 0:
        shift_seconds = tf.random.uniform((), minval=-float(max_t_shift), maxval=float(max_t_shift))
        shift_samples = tf.cast(tf.round(shift_seconds * sampling_rate), tf.int32)
        x1 = tf.roll(x1, shift=shift_samples, axis=1)
        x2 = tf.roll(x2, shift=shift_samples, axis=1)
        x3 = tf.roll(x3, shift=shift_samples, axis=1)
        x4 = tf.roll(x4, shift=shift_samples, axis=1)

    # Optional trim (center crop) along time axis
    if trim_sec and trim_sec > 0:
        trim = tf.cast(tf.round(trim_sec * sampling_rate), tf.int32)
        T = tf.shape(x1)[1]
        # only trim if we have room on both sides
        can_trim = tf.greater(T, 2 * trim)
        def do_trim(t):
            return tf.gather(t, tf.range(trim, T - trim), axis=1)
        x1 = tf.cond(can_trim, lambda: do_trim(x1), lambda: x1)
        x2 = tf.cond(can_trim, lambda: do_trim(x2), lambda: x2)
        x3 = tf.cond(can_trim, lambda: do_trim(x3), lambda: x3)
        x4 = tf.cond(can_trim, lambda: do_trim(x4), lambda: x4)

    return x1, x2, x3, x4

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

class ExpandDims1(tf.keras.layers.Layer): # this is for envelope decoder
    def call(self, inputs):
        return tf.expand_dims(inputs, axis=1)

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

# The transformer architecture (no dropout)
class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(embed_dim)
        ])
        
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, dropout=False, mask=None, return_attention=False):
        attn_output, attn_weights = self.att(
            inputs, inputs, attention_mask=mask, return_attention_scores=True
        )  # <- Extract attention scores
        
        out1 = self.layernorm1(inputs + attn_output)
        out1 = self.dropout1(out1, training=dropout)
        
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + ffn_output)
        out2 = self.dropout2(out2, training=dropout)

        if return_attention == True:
            return out2, attn_weights  # Return both outputs and attention weights
        else:
            return out2

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

# Generate clustering layer
@tf.keras.utils.register_keras_serializable(package="infra")
class ClusteringLayer(layers.Layer):
    """
    DEC-style soft assignment head.

    Inputs:
      z: [B, D] (or [B, T, D] if fuse_mode is used)

    Args:
      n_clusters: number of clusters K
      alpha: Student-t dof (DEC default 1.0)
      temperature: distance scaling (>1.0 softens, <1.0 sharpens)
      fuse_mode: None | "mean" | "flatten"
      use_l2_norm: if True, L2-normalize z & centroids (cosine-like).
                   DEFAULT False → magnitude-sensitive Euclidean.
    """
    def __init__(self,
                 n_clusters,
                 alpha=1.0,
                 temperature=1.0,
                 fuse_mode=None,
                 use_l2_norm=False,   # <-- magnitude-sensitive by default
                 **kwargs):
        super().__init__(**kwargs)
        self.n_clusters  = int(n_clusters)
        self.alpha       = float(alpha)
        self.temperature = float(temperature)
        self.fuse_mode   = fuse_mode
        self.use_l2_norm = bool(use_l2_norm)
        self._latent_dim = None

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(
            n_clusters=self.n_clusters,
            alpha=self.alpha,
            temperature=self.temperature,
            fuse_mode=self.fuse_mode,
            use_l2_norm=self.use_l2_norm,
        ))
        return cfg

    def build(self, input_shape):
        last_dim = int(input_shape[-1])
        self._latent_dim = last_dim
        self.centroids = self.add_weight(
            name=f"{self.name}_centroids",
            shape=(self.n_clusters, self._latent_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def _fuse(self, z):
        if z.shape.rank == 3 and self.fuse_mode:
            if self.fuse_mode == "mean":
                z = tf.reduce_mean(z, axis=1)              # [B, D]
            elif self.fuse_mode == "flatten":
                z = tf.reshape(z, [tf.shape(z)[0], -1])     # [B, T*D]
            else:
                raise ValueError(f"Unsupported fuse_mode: {self.fuse_mode}")
        return z

    def call(self, z):
        eps = 1e-8
        z = self._fuse(z)                                   # [B, D]

        c = self.centroids
        if self.use_l2_norm:                                # optional cosine-like mode
            z = tf.math.l2_normalize(z, axis=1)
            c = tf.math.l2_normalize(c, axis=1)

        # Squared Euclidean distances
        zi = tf.expand_dims(z, 1)                           # [B,1,D]
        ck = tf.expand_dims(c, 0)                           # [1,K,D]
        d2 = tf.reduce_sum(tf.square(zi - ck), axis=2)      # [B,K]
        if self.temperature != 1.0:
            d2 = d2 / self.temperature

        # Student-t kernel (DEC)
        power = -(self.alpha + 1.0) / 2.0
        q = tf.pow(1.0 + d2 / tf.maximum(self.alpha, eps), power)   # [B,K]
        q = q / tf.reduce_sum(q, axis=1, keepdims=True)
        return tf.clip_by_value(q, eps, 1.0)

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def build_InfraCoder(num_freq_samples, num_time_samples, K_s=6, K_e=3, K_p=6, lr=1e-3, l2_lambda=1e-4, return_attention=True):
    # Define optimizer
    adam = Adam(learning_rate=lr, epsilon=1e-6)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Define shape of scalogram input [64,800,1]
    input_tensor_scalogram = Input(shape=(num_freq_samples, num_time_samples, 1), name='Input_Scalogram', dtype=tf.float32)
    def scalogram_encoder(input_tensor_scalogram, return_attention=return_attention): # latent_space_size=latent_space_size):
        # Encoder 1
        # Conv 1 - Dilated Stride block
        x_s = Conv2D(4,
                     kernel_size=(3,3),
                     strides=(1,1),
                     dilation_rate=3, # now [7,7]
                     padding='same',
                     kernel_initializer='he_uniform',
                     name='Conv_1a_Scalogram')(input_tensor_scalogram)
        x_s = BatchNormalization()(x_s)
        x_s = activations.relu(x_s)
        x_s = Conv2D(8,
                   kernel_size=(3,3), # keep T kernel dim at 3 for strided layers to avoid onset blurring
                   strides=(2,2), # [32, 400]
                   padding='same',
                   kernel_initializer='he_uniform', 
                   name='Conv_1b_Scalogram')(x_s)
        #-----------------------------------------------------------------------------------------------------------------------#
        # Conv 2 - Resblock
        x_res = BatchNormalization()(x_s)
        x_res = activations.relu(x_res)
        x_res = Conv2D(8,
                       kernel_size=(3,5),
                       strides=(1,1), 
                       padding='same',
                       kernel_initializer='he_uniform', 
                       name='Conv_2a_Res_Scalogram')(x_res)
        x_res = BatchNormalization()(x_res)
        x_res = activations.relu(x_res)
        x_res = Conv2D(8,
                       kernel_size=(3,5),
                       strides=(1,1), 
                       padding='same',
                       kernel_initializer='he_uniform', 
                       name='Conv_2b_Res_Scalogram')(x_res)
        out = Add()([x_s, x_res])
        x_s = BatchNormalization()(out)
        x_s = activations.relu(x_s)
        #-----------------------------------------------------------------------------------------------------------------------#
        # Conv 3 - Stacked Stride block
        x_s = Conv2D(16,
                     kernel_size=(3,3),
                     strides=(2,1), # [16, 400]
                     padding='same',
                     kernel_initializer='he_uniform', 
                     name='Conv_3a_Scalogram')(x_s)
        x_s = BatchNormalization()(x_s)
        x_s = activations.relu(x_s)
        x_s = Conv2D(16,
                     kernel_size=(3,3),
                     strides=(2,2), # [8, 200]
                     padding='same',
                     kernel_initializer='he_uniform', 
                     name='Conv_3b_Scalogram')(x_s)
        x_s = BatchNormalization()(x_s)
        x_s = activations.relu(x_s)
        #-----------------------------------------------------------------------------------------------------------------------#
        # Conv 4 - Final Conv layer
        x_s = Conv2D(32,
                     kernel_size=(3,5),
                     strides=(1,1), # [8, 200]
                     padding='same',
                     kernel_initializer='he_uniform', 
                     name='Conv_4_Scalogram')(x_s)
        x_s = BatchNormalization()(x_s)
        x_s = activations.relu(x_s)
        #-----------------------------------------------------------------------------------------------------------------------#
        # Pre-Transformer: learned frequency pooling
        # x: [B, F=8, T=200, C=32]
        q = Conv2D(4, (3,1), activation='tanh', padding='same', name='Scalogram_FreqAttn_Conv')(x_s)
        w = Dense(1, name='Scalogram_FreqAttn_Score')(q)
        a = Softmax(axis=1, name='Scalogram_FreqAttn_Softmax')(w)
        x_s_attn = layers.Lambda(lambda t: keras.ops.sum(t[0] * t[1], axis=1), name="Scalogram_FreqAttn_Pool")([x_s, a]) # [B, 100, 32]
        # normalize tokens before transformer
        x_s_attn = LayerNormalization(epsilon=1e-6, name='Scalogram_PreTF_LN')(x_s_attn)
        #-----------------------------------------------------------------------------------------------------------------------#
        # Transformer block
        if return_attention == True:
            x_s, scalogram_attention = TransformerBlock(embed_dim=32, num_heads=4, ff_dim=64)(x_s_attn, mask=None, return_attention=return_attention)
        else:
            x_s = TransformerBlock(embed_dim=32, num_heads=4, ff_dim=64)(x_s_attn, mask=None, return_attention=return_attention) 
            scalogram_attention = None
        #-----------------------------------------------------------------------------------------------------------------------#
        # Return attention map
        return x_s, scalogram_attention
    #-----------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------------------------------------------------------------------------#
    # Define shape of Envelope input [1,800,1]
    input_tensor_envelope = Input(shape=(1, num_time_samples, 1), name='Input_Envelope', dtype=tf.float32)
    def envelope_encoder(input_tensor_envelope, return_attention=return_attention): #latent_space_size=latent_space_size):
        # Encoder 2
        input_tensor_envelope = np.squeeze(input_tensor_envelope, axis=1)
        # Conv 1 - Dilated stride block
        x_h = Conv1D(4,
                     kernel_size=3,
                     strides=1, 
                     dilation_rate=5, # now 9
                     padding='same',
                     kernel_initializer='he_uniform', 
                     name='Conv_1a_Hilbert')(input_tensor_envelope)
        x_h = BatchNormalization()(x_h)
        x_h = activations.relu(x_h)
        x_h = Conv1D(8,
                     kernel_size=7,
                     strides=2, # [400]
                     padding='same',
                     kernel_initializer='he_uniform', 
                     name='Conv_1b_Hilbert')(x_h)
        #-----------------------------------------------------------------------------------------------------------------------#
        # Conv 2 - Resblock
        x_res = BatchNormalization()(x_h)
        x_res = activations.relu(x_res)
        x_res = Conv1D(8,
                       kernel_size=7,
                       strides=1, 
                       padding='same',
                       kernel_initializer='he_uniform', 
                       name='Conv_2a_Res_Hilbert')(x_res)
        x_res = BatchNormalization()(x_res)
        x_res = activations.relu(x_res)
        x_res = Conv1D(8,
                       kernel_size=7,
                       strides=1, 
                       padding='same',
                       kernel_initializer='he_uniform', 
                       name='Conv_2b_Res_Hilbert')(x_res)
        out = Add()([x_h, x_res])
        x_h = BatchNormalization()(out)
        x_h = activations.relu(x_h)
        #-----------------------------------------------------------------------------------------------------------------------#
        # Conv 3 - Stacked Stride block
        x_h = Conv1D(16,
                     kernel_size=7,
                     strides=1, # [400]
                     padding='same',
                     kernel_initializer='he_uniform', 
                     name='Conv_3a_Hilbert')(x_h)
        x_h = BatchNormalization()(x_h)
        x_h = activations.relu(x_h)
        x_h = Conv1D(16,
                     kernel_size=7,
                     strides=2, # [200]
                     padding='same',
                     kernel_initializer='he_uniform', 
                     name='Conv_3b_Hilbert')(x_h)
        x_h = BatchNormalization()(x_h)
        x_h = activations.relu(x_h)
        #-----------------------------------------------------------------------------------------------------------------------#
        # Conv 4 - Final Conv layer
        x_h = Conv1D(32,
                     kernel_size=7,
                     strides=1, # [200]
                     padding='same',
                     kernel_initializer='he_uniform', 
                     name='Conv_4_Hilbert')(x_h)
        x_h = BatchNormalization()(x_h)
        x_h = activations.relu(x_h)
        #-----------------------------------------------------------------------------------------------------------------------#
        # Transformer block
        x_h = LayerNormalization(epsilon=1e-6, name="Envelope_PreTF_LN")(x_h)
        if return_attention == True:
            x_h, envelope_attention = TransformerBlock(embed_dim=32, num_heads=4, ff_dim=64)(x_h, mask=None, return_attention=return_attention)
        else:
            x_h = TransformerBlock(embed_dim=32, num_heads=4, ff_dim=64)(x_h, mask=None, return_attention=return_attention) 
            envelope_attention = None
        #-----------------------------------------------------------------------------------------------------------------------#
        # Return attention map
        return x_h, envelope_attention
    #-----------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------------------------------------------------------------------------#
    # Define shape of Phase input [64,800,2]
    # Concatenating inside the model
    input_tensor_sin = Input(shape=(num_freq_samples, num_time_samples, 1), name='Input_Phase_Sine', dtype=tf.float32)
    input_tensor_cos = Input(shape=(num_freq_samples, num_time_samples, 1), name='Input_Phase_Cosine', dtype=tf.float32)
    def phase_encoder(input_tensor_sin, input_tensor_cos, return_attention=return_attention): # latent_space_size=latent_space_size): 
        # Encoder 3
        # Concatenate sine and cosine channel-wise
        x_p = concatenate([input_tensor_sin, input_tensor_cos], axis=-1)
        #-----------------------------------------------------------------------------------------------------------------------#
        # Conv 1 Dilated Stride block
        x_p = Conv2D(4,
                     kernel_size=(3,3),
                     strides=(1,1),
                     dilation_rate=3, # now [7,7]
                     padding='same',
                     kernel_initializer='he_uniform', 
                     name='Conv_1a_Phase')(x_p)
        x_p = BatchNormalization()(x_p)
        x_p = activations.relu(x_p)
        x_p = Conv2D(8,
                     kernel_size=(3,3), # keep T kernel dim at 3 for strided layers to avoid onset blurring
                     strides=(2,2), # [32, 400]
                     padding='same',
                     kernel_initializer='he_uniform', 
                     name='Conv_1b_Phase')(x_p)
        #-----------------------------------------------------------------------------------------------------------------------#
        # Conv 2 - Resblock
        x_res = BatchNormalization()(x_p)
        x_res = activations.relu(x_res)
        x_res = Conv2D(8,
                       kernel_size=(3,5),
                       strides=(1,1), 
                       padding='same',
                       kernel_initializer='he_uniform', 
                       name='Conv_2a_Res_Phase')(x_res)
        x_res = BatchNormalization()(x_res)
        x_res = activations.relu(x_res)
        x_res = Conv2D(8,
                       kernel_size=(3,5),
                       strides=(1,1), 
                       padding='same',
                       kernel_initializer='he_uniform', 
                       name='Conv_1b_Res_Phase')(x_res)
        out = Add()([x_p, x_res])
        x_p = BatchNormalization()(out)
        x_p = activations.relu(x_p)
        #-----------------------------------------------------------------------------------------------------------------------#
        # Conv 3 - Stacked Stride block
        x_p = Conv2D(16,
                     kernel_size=(3,3),
                     strides=(2,1), # [16, 400]
                     padding='same',
                     kernel_initializer='he_uniform', 
                     name='Conv_3a_Phase')(x_p)
        x_p = BatchNormalization()(x_p)
        x_p = activations.relu(x_p)
        x_p = Conv2D(16,
                     kernel_size=(3,3),
                     strides=(2,2), # [8, 200]
                     padding='same',
                     kernel_initializer='he_uniform',
                     name='Conv_3b_Phase')(x_p)
        x_p = BatchNormalization()(x_p)
        x_p = activations.relu(x_p)
        #-----------------------------------------------------------------------------------------------------------------------#
        # Conv 4 - Final Conv layer
        x_p = Conv2D(32,
                     kernel_size=(3,5),
                     strides=(1,1), # [8, 200]
                     padding='same',
                     kernel_initializer='he_uniform', 
                     name='Conv_4_Phase')(x_p)
        x_p = BatchNormalization()(x_p)
        x_p = activations.relu(x_p)
        #-----------------------------------------------------------------------------------------------------------------------#
        # Pre-Transformer: learned frequency pooling
        # x: [B, F=8, T=200, C=32]
        q = Conv2D(4, (3,1), activation='tanh', padding='same', name='Phase_FreqAttn_Conv')(x_p)
        w = Dense(1, name='Phase_FreqAttn_Score')(q)
        a = Softmax(axis=1, name='Phase_FreqAttn_Softmax')(w)
        x_p_attn = layers.Lambda(lambda t: keras.ops.sum(t[0] * t[1], axis=1), name="Phase_FreqAttn_Pool")([x_p, a]) # [B, 100, 32]
        # normalize tokens before transformer
        x_p_attn = LayerNormalization(epsilon=1e-6, name='Phase_PreTF_LN')(x_p_attn)
        #-----------------------------------------------------------------------------------------------------------------------#
        # Transformer block
        if return_attention == True:
            x_p, phase_attention = TransformerBlock(embed_dim=32, num_heads=4, ff_dim=64)(x_p_attn, mask=None, return_attention=return_attention)
        else:
            x_p = TransformerBlock(embed_dim=32, num_heads=4, ff_dim=64)(x_p_attn, mask=None, return_attention=return_attention)
            phase_attention = None
        #-----------------------------------------------------------------------------------------------------------------------#
        # Return attention map
        return x_p, phase_attention
    #-----------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------------------------------------------------------------------------#
    # Latent space
    # Current calls (unchanged signatures - transformer outputs)
    enc1_map, scalogram_attention = scalogram_encoder(input_tensor_scalogram, return_attention=return_attention) # [200, 32]
    enc2_map, envelope_attention = envelope_encoder(input_tensor_envelope, return_attention=return_attention) # [200, 32]
    enc3_map, phase_attention = phase_encoder(input_tensor_sin, input_tensor_cos, return_attention=return_attention) # [200, 32]
    #-----------------------------------------------------------------------------------------------------------------------#
    # Per-modality latent space
    def channel_bottleneck(x, K, name):
        # x: [B, 200, 32]
        x = Conv1D(K, kernel_size=1, padding='same', use_bias=True, 
                   kernel_initializer='he_uniform', name=f'{name}_proj1x1')(x) # linear projection 
        x = LayerNormalization(epsilon=1e-6, name=f'{name}_LN')(x)
        return x
    # enc*_map: [B,100,32] from each encoder’s Transformer
    # K_s = 6; K_e = 3; K_p = 6
    g1 = channel_bottleneck(enc1_map, K=K_s, name='scalo') # [B,200,K_s]
    g1_dec = activations.relu(g1)
    g2 = channel_bottleneck(enc2_map, K=K_e, name='env') # [B,200,K_e]
    g2_dec = activations.relu(g2) 
    g3 = channel_bottleneck(enc3_map, K=K_p, name='phase') # [B,200,K_p]
    g3_dec = activations.gelu(g3) 
    #-----------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------------------------------------------------------------------------#
    # Scalogram Decoder
    # Stacked Conv - Upsampling block 1
    x_s_decoder = Reshape((1,200,K_s), name='Input_Scalogram_Decoder')(g1_dec)
    x_s_decoder = Conv2D(4, 
                         kernel_size=(3, 5),
                         padding='same',
                         kernel_initializer='he_uniform',
                         name='Decoder_Conv_1a_Scalogram')(x_s_decoder)
    x_s_decoder = LayerNormalization()(x_s_decoder)
    x_s_decoder = activations.gelu(x_s_decoder)
    x_s_decoder = UpSampling2D(size=(2,1), # [2,200]
                               interpolation='bilinear',
                               name='Upsample_1a_Scalogram')(x_s_decoder) 
    x_s_decoder = Conv2D(8, 
                         kernel_size=(3, 5),
                         padding='same',
                         kernel_initializer='he_uniform',
                         name='Decoder_Conv_1b_Scalogram')(x_s_decoder)
    x_s_decoder = LayerNormalization()(x_s_decoder)
    x_s_decoder = activations.gelu(x_s_decoder)
    x_s_decoder = UpSampling2D(size=(4,1), # [8,200]
                               interpolation='bilinear',
                               name='Upsample_1b_Scalogram')(x_s_decoder) 
    #-----------------------------------------------------------------------------------------------------------------------#
    # Conv - Upsampling 2
    x_s_decoder = Conv2D(8, 
                         kernel_size=(3, 5),
                         padding='same',
                         kernel_initializer='he_uniform', 
                         name='Decoder_Conv_2a_Scalogram')(x_s_decoder)
    x_s_decoder = LayerNormalization()(x_s_decoder)
    x_s_decoder = activations.gelu(x_s_decoder)
    x_s_decoder = UpSampling2D(size=(2,2), # [16,400]
                               interpolation='bilinear',
                               name='Upsample_2_Scalogram')(x_s_decoder) 
    x_s_decoder = Conv2D(16, 
                         kernel_size=(3,3),
                         padding='same',
                         kernel_initializer='he_uniform', 
                         name='Decoder_Conv_2b_Scalogram')(x_s_decoder)
    x_s_decoder = LayerNormalization()(x_s_decoder)
    x_s_decoder = activations.gelu(x_s_decoder)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Conv - Upsampling Resblock 3
    x_s_decoder_res = Conv2D(16, 
                             kernel_size=(3, 3), 
                             padding='same',
                             kernel_initializer='he_uniform',
                             name='Decoder_Conv_3a_Scalogram')(x_s_decoder)
    x_s_decoder_res = LayerNormalization()(x_s_decoder_res)
    x_s_decoder_res = activations.gelu(x_s_decoder_res)
    x_s_decoder_res = Conv2D(16, 
                             kernel_size=(3, 3), 
                             padding='same',
                             kernel_initializer='he_uniform', 
                             name='Decoder_Conv_3b_Scalogram')(x_s_decoder_res)
    out = Add()([x_s_decoder, x_s_decoder_res])
    x_s_decoder = LayerNormalization()(out)
    x_s_decoder = activations.gelu(x_s_decoder)
    x_s_decoder = UpSampling2D(size=(2,2), # [32,800]
                               interpolation='bilinear',
                               name='Upsample_3_Scalogram')(x_s_decoder) 
    #-----------------------------------------------------------------------------------------------------------------------#
    # Conv - Upsampling 4
    x_s_decoder = Conv2D(32, 
                         kernel_size=(3, 3),
                         padding='same',
                         kernel_initializer='he_uniform', 
                         name='Decoder_Conv_4_Scalogram')(x_s_decoder)
    x_s_decoder = LayerNormalization()(x_s_decoder)
    x_s_decoder = activations.gelu(x_s_decoder)
    x_s_decoder = UpSampling2D(size=(2,1), # [64,800]
                               interpolation='bilinear',
                               name='Upsample_4_Scalogram')(x_s_decoder) 
    # Adding residual block
    x_s_decoder_res = Conv2D(32, 
                         kernel_size=(3, 3),
                         padding='same',
                         kernel_initializer='he_uniform', 
                         name='Decoder_Conv_4a_ResScalogram')(x_s_decoder)
    x_s_decoder_res = LayerNormalization()(x_s_decoder_res)
    x_s_decoder_res = activations.gelu(x_s_decoder_res)
    x_s_decoder_res = Conv2D(32, 
                         kernel_size=(3, 3),
                         padding='same',
                         kernel_initializer='he_uniform', 
                         name='Decoder_Conv_4b_ResScalogram')(x_s_decoder_res)
    out = Add()([x_s_decoder, x_s_decoder_res])
    x_s_decoder = LayerNormalization()(out)
    x_s_decoder = activations.gelu(x_s_decoder)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Map it to single output channel
    decoded_scalogram = Conv2D(1, 
                               (3,3),
                               padding='same',
                               name='Output_Scalogram')(x_s_decoder)
    decoded_scalogram = activations.linear(decoded_scalogram)
    #-----------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------------------------------------------------------------------------#
    # Envelope Decoder
    # Conv - Upsampling 1
    x_h_decoder = Reshape((200,K_e), name='Input_Envelope_Decoder')(g2_dec) 
    x_h_decoder = Conv1D(4, 
                         kernel_size=7,
                         padding='same',
                         kernel_initializer='he_uniform',
                         name='Decoder_Conv_1a_Envelope')(x_h_decoder)
    x_h_decoder = LayerNormalization()(x_h_decoder)
    x_h_decoder = activations.gelu(x_h_decoder)
    x_h_decoder = UpSampling1D(size=2, # [400]
                               name='Upsample_1_Envelope')(x_h_decoder) 
    x_h_decoder = Conv1D(8, 
                         kernel_size=7,
                         padding='same',
                         kernel_initializer='he_uniform',
                         name='Decoder_Conv_1b_Envelope')(x_h_decoder)
    x_h_decoder = LayerNormalization()(x_h_decoder)
    x_h_decoder = activations.gelu(x_h_decoder)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Conv 2
    x_h_decoder = Conv1D(16, 
                         kernel_size=5,
                         padding='same',
                         kernel_initializer='he_uniform', 
                         name='Decoder_Conv_2_Envelope')(x_h_decoder)
    x_h_decoder = LayerNormalization()(x_h_decoder)
    x_h_decoder = activations.gelu(x_h_decoder)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Conv - Upsampling Resblock 3
    x_h_decoder_res = Conv1D(16, 
                             kernel_size=5, 
                             padding='same',
                             kernel_initializer='he_uniform',
                             name='Decoder_Conv_3a_Envelope')(x_h_decoder)
    x_h_decoder_res = LayerNormalization()(x_h_decoder_res)
    x_h_decoder_res = activations.gelu(x_h_decoder_res)
    x_h_decoder_res = Conv1D(16, 
                             kernel_size=5, 
                             padding='same',
                             kernel_initializer='he_uniform', 
                             name='Decoder_Conv_3b_Envelope')(x_h_decoder_res)
    out = Add()([x_h_decoder, x_h_decoder_res])
    x_h_decoder = LayerNormalization()(out)
    x_h_decoder = UpSampling1D(size=2, # [800]
                               name='Upsample_3_Envelope')(x_h_decoder) 
    #-----------------------------------------------------------------------------------------------------------------------#
    # Conv 4
    x_h_decoder = Conv1D(32, 
                         kernel_size=3,
                         padding='same',
                         kernel_initializer='he_uniform', 
                         name='Decoder_Conv_4_Envelope')(x_h_decoder)
    x_h_decoder = LayerNormalization()(x_h_decoder)
    x_h_decoder = activations.gelu(x_h_decoder)
    # Adding residual block
    x_h_decoder_res = Conv1D(32, 
                         kernel_size=3,
                         padding='same',
                         kernel_initializer='he_uniform', 
                         name='Decoder_Conv_4a_ResEnvelope')(x_h_decoder)
    x_h_decoder_res = LayerNormalization()(x_h_decoder_res)
    x_h_decoder_res = activations.gelu(x_h_decoder_res)
    x_h_decoder_res = Conv1D(32, 
                         kernel_size=3,
                         padding='same',
                         kernel_initializer='he_uniform', 
                         name='Decoder_Conv_4b_ResEnvelope')(x_h_decoder_res)
    out = Add()([x_h_decoder, x_h_decoder_res])
    x_h_decoder = LayerNormalization()(out)
    x_h_decoder = activations.gelu(x_h_decoder)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Map it to single output channel
    decoded_envelope = Conv1D(1, 
                              kernel_size=3,
                              padding='same')(x_h_decoder)
    decoded_envelope = activations.linear(decoded_envelope)
    decoded_envelope = ExpandDims1(name="Output_Envelope")(decoded_envelope)
    #-----------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------------------------------------------------------------------------#
    # Phase Decoder
    # Conv - Upsampling 1
    x_p_decoder = Reshape((1,200,K_p), name='Input_Phase_Decoder')(g3_dec)
    x_p_decoder = Conv2D(4,
                         (3,5),
                         padding='same',
                         kernel_initializer='he_uniform',
                         name='Decoder_Conv_1a_Phase')(x_p_decoder)
    x_p_decoder = LayerNormalization()(x_p_decoder)
    x_p_decoder = activations.gelu(x_p_decoder)
    x_p_decoder = UpSampling2D(size=(2,1), # [2,200]
                               interpolation='bilinear',
                               name='Upsample_1a_Phase')(x_p_decoder) 
    x_p_decoder = Conv2D(8, 
                         (3,5),
                         padding='same',
                         kernel_initializer='he_uniform',
                         name='Decoder_Conv_1b_Phase')(x_p_decoder)
    x_p_decoder = LayerNormalization()(x_p_decoder)
    x_p_decoder = activations.gelu(x_p_decoder)
    x_p_decoder = UpSampling2D(size=(4,1), # [8,200]
                               interpolation='bilinear',
                               name='Upsample_1b_Phase')(x_p_decoder) 
    #-----------------------------------------------------------------------------------------------------------------------#
    # Conv - Upsampling 2
    x_p_decoder = Conv2D(8, 
                         (3,5),
                         padding='same',
                         kernel_initializer='he_uniform', 
                         name='Decoder_Conv_2a_Phase')(x_p_decoder)
    x_p_decoder = LayerNormalization()(x_p_decoder)
    x_p_decoder = activations.gelu(x_p_decoder)
    x_p_decoder = UpSampling2D(size=(2,2), # [16,400]
                               interpolation='bilinear',
                               name='Upsample_2_Phase')(x_p_decoder) 
    x_p_decoder = Conv2D(16, 
                         (3,3),
                         padding='same',
                         kernel_initializer='he_uniform', 
                         name='Decoder_Conv_2b_Phase')(x_p_decoder)
    x_p_decoder = LayerNormalization()(x_p_decoder)
    x_p_decoder = activations.gelu(x_p_decoder)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Conv - Upsample Resblock 3
    x_p_decoder_res = Conv2D(16, 
                             (3,3), 
                             padding='same',
                             kernel_initializer='he_uniform',
                             name='Decoder_Conv_3a_Phase')(x_p_decoder)
    x_p_decoder_res = LayerNormalization()(x_p_decoder_res)
    x_p_decoder_res = activations.gelu(x_p_decoder_res)
    x_p_decoder_res = Conv2D(16, 
                             (3,3), 
                             padding='same',
                             kernel_initializer='he_uniform', 
                             name='Decoder_Conv_3b_Phase')(x_p_decoder_res)
    out = Add()([x_p_decoder, x_p_decoder_res])
    x_p_decoder = LayerNormalization()(out)
    x_p_decoder = activations.gelu(x_p_decoder)
    x_p_decoder = UpSampling2D(size=(2,2), # [32,800]
                               interpolation='bilinear',
                               name='Upsample_3_Phase')(x_p_decoder) 
    #-----------------------------------------------------------------------------------------------------------------------#
    # Conv 4 Sine
    x_p_sin_decoder = Conv2D(32, 
                         (3,3),
                         padding='same',
                         kernel_initializer='he_uniform', 
                         name='Decoder_Conv_4a_Phase_Sine')(x_p_decoder)
    x_p_sin_decoder = LayerNormalization()(x_p_sin_decoder)
    x_p_sin_decoder = activations.gelu(x_p_sin_decoder)
    x_p_sin_decoder = UpSampling2D(size=(2,1), # [64,800]
                               interpolation='bilinear',
                               name='Upsample_4_Phase_Sine')(x_p_sin_decoder) 
    # Adding residual block
    x_p_sin_decoder_res = Conv2D(32, 
                         kernel_size=(3, 3),
                         padding='same',
                         kernel_initializer='he_uniform', 
                         name='Decoder_Conv_4a_ResPhase_Sine')(x_p_sin_decoder)
    x_p_sin_decoder_res = LayerNormalization()(x_p_sin_decoder_res)
    x_p_sin_decoder_res = activations.gelu(x_p_sin_decoder_res)
    x_p_sin_decoder_res = Conv2D(32, 
                         kernel_size=(3, 3),
                         padding='same',
                         kernel_initializer='he_uniform', 
                         name='Decoder_Conv_4b_ResPhase_Sine')(x_p_sin_decoder_res)
    out = Add()([x_p_sin_decoder, x_p_sin_decoder_res])
    x_p_sin_decoder = LayerNormalization()(out)
    x_p_sin_decoder = activations.gelu(x_p_sin_decoder)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Map it to single output channel (sine components)
    decoded_sin = Conv2D(1, 
                           (3,3),
                           padding='same',
                           name='Output_Phase_Sine')(x_p_sin_decoder)
    decoded_sin = activations.linear(decoded_sin)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Conv 4 Cosine
    x_p_cos_decoder = Conv2D(32, 
                         (3,3),
                         padding='same',
                         kernel_initializer='he_uniform', 
                         name='Decoder_Conv_4a_Phase_Cosine')(x_p_decoder)
    x_p_cos_decoder = LayerNormalization()(x_p_cos_decoder)
    x_p_cos_decoder = activations.gelu(x_p_cos_decoder)
    x_p_cos_decoder = UpSampling2D(size=(2,1), # [64,800]
                               interpolation='bilinear',
                               name='Upsample_4_Phase_Cosine')(x_p_cos_decoder) 
    # Adding residual block
    x_p_cos_decoder_res = Conv2D(32, 
                         kernel_size=(3, 3),
                         padding='same',
                         kernel_initializer='he_uniform', 
                         name='Decoder_Conv_4a_ResPhase_Cosine')(x_p_cos_decoder)
    x_p_cos_decoder_res = LayerNormalization()(x_p_cos_decoder_res)
    x_p_cos_decoder_res = activations.gelu(x_p_cos_decoder_res)
    x_p_cos_decoder_res = Conv2D(32, 
                         kernel_size=(3, 3),
                         padding='same',
                         kernel_initializer='he_uniform', 
                         name='Decoder_Conv_4b_ResPhase_Cosine')(x_p_cos_decoder_res)
    out = Add()([x_p_cos_decoder, x_p_cos_decoder_res])
    x_p_cos_decoder = LayerNormalization()(out)
    x_p_cos_decoder = activations.gelu(x_p_cos_decoder)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Map it to single output channel (sine components)
    decoded_cos = Conv2D(1, 
                           (3,3),
                           padding='same',
                           name='Output_Phase_Cosine')(x_p_cos_decoder)
    decoded_cos = activations.linear(decoded_cos)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Merge to 2 channels and force unit length per pixel to apply phase angle loss function
    phase_logits = Concatenate(axis=-1, name='phase_logits')([decoded_sin, decoded_cos])  # [B,64,800,2]
    decoded_phase = UnitNormalization(axis=-1, name='Recon_Phase')(phase_logits)
    #-----------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------------------------------------------------------------------------#
    # Define Autoencoder 
    # TRAINING MODEL: only recon outputs
    infracoder = tf.keras.Model(
        inputs=[input_tensor_scalogram, input_tensor_envelope, input_tensor_sin, input_tensor_cos],
        outputs=[decoded_scalogram, decoded_envelope, decoded_phase],
        name='Infracoder')
    # Optional: separate attention model for inference
    if return_attention:
        attention_model = tf.keras.Model(
            inputs=infracoder.inputs,
            outputs=[scalogram_attention, envelope_attention, phase_attention],
            name='InfraCoder_Attention'
        )
    else:
        attention_model = None
    #-----------------------------------------------------------------------------------------------------------------------#
    # Compile: one loss per output (order must match outputs)
    infracoder.compile(
        optimizer=adam,
        loss=[tf.keras.losses.MeanSquaredError(), # scalogram
              tf.keras.losses.MeanSquaredError(), # envelope
              phase_angle_loss], # phase
        loss_weights=[1.0, # scalogram
                      1.0, # envelope
                      1.0] # phase
    )
    #-----------------------------------------------------------------------------------------------------------------------#
    # Return models
    if attention_model is not None:
        return infracoder, attention_model
    else:
        return infracoder

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

# -------- Loss (per-batch mean of 1 - cosΔ): fine for training --------
@tf.keras.utils.register_keras_serializable(package="infra")
def phase_angle_loss(y_true, y_pred, eps=1e-6):
    # Ensure we're comparing directions only
    y_true = y_true / (tf.norm(y_true, axis=-1, keepdims=True) + eps)
    y_pred = y_pred / (tf.norm(y_pred, axis=-1, keepdims=True) + eps)
    cos_d  = tf.reduce_sum(y_true * y_pred, axis=-1)        # [B,H,W]
    return tf.reduce_mean(1.0 - cos_d)                      # batch mean

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

# ---------- Build a non-augmented eval dataset ----------
def build_eval_ds(X_scalo, X_env, X_sin, X_cos, batch_size=32, target_T=800):
    base = tf.data.Dataset.from_tensor_slices((
        X_scalo.astype('float32'),    # (N, 64, T, [1]?) or (N, 64, T)
        X_env.astype('float32'),      # (N,  1, T, [1]?) or (N,  1, T)
        X_sin.astype('float32'),      # (N, 64, T, [1]?) or (N, 64, T)
        X_cos.astype('float32'),      # (N, 64, T, [1]?) or (N, 64, T)
    ))

    def ensure_3d(x):
        # If rank==2 (F, T), add channel dim -> (F, T, 1)
        return tf.cond(
            tf.equal(tf.rank(x), 2),
            lambda: tf.expand_dims(x, axis=-1),
            lambda: x
        )

    def center_crop_time(x, target_T):
        """
        Make x shape (F, T, C); crop along time axis to target_T.
        If T <= target_T, return x unchanged (no padding here).
        """
        x = ensure_3d(x)                  # (F, T, C)
        T = tf.shape(x)[1]

        def _crop():
            start = (T - target_T) // 2
            return x[:, start:start + target_T, :]

        return tf.cond(T > target_T, _crop, lambda: x)

    def pack_io(x1, x2, x3, x4):
        # Center-crop time to 800
        x1 = center_crop_time(x1, target_T)   # scalogram: (64, 800, 1)
        x2 = center_crop_time(x2, target_T)   # envelope:  ( 1, 800, 1)
        x3 = center_crop_time(x3, target_T)   # phase sin: (64, 800, 1)
        x4 = center_crop_time(x4, target_T)   # phase cos: (64, 800, 1)

        # Set static shapes (helps model build)
        x1 = tf.ensure_shape(x1, (64, 800, 1))
        x2 = tf.ensure_shape(x2, (1, 800, 1))
        x3 = tf.ensure_shape(x3, (64, 800, 1))
        x4 = tf.ensure_shape(x4, (64, 800, 1))

        # Targets: phase is 2 channels concatenated
        y_phase = tf.concat([x3, x4], axis=-1)   # (64, 800, 2)
        inputs  = (x1, x2, x3, x4)
        targets = (x1, x2, y_phase)
        return inputs, targets

    return (base
            .map(pack_io, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size, drop_remainder=False)
            .prefetch(tf.data.AUTOTUNE))

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

# ---------- One-pass evaluator (micro averages) ----------
def evaluate_offline_metrics(model, ds, eps=1e-6):
    """
    Assumes model outputs = [scalo_out, env_out, phase_out],
    where phase_out is 2-channel (sin, cos).
    Returns dict with r2_scalo, r2_env, mean_angle_deg, mean_cos, angular_loss.
    """
    # Accumulators for R²: SSE and SStot pieces
    sse_scal = 0.0; sumy_scal = 0.0; sumy2_scal = 0.0; n_scal = 0
    sse_env  = 0.0; sumy_env  = 0.0; sumy2_env  = 0.0; n_env  = 0

    # Accumulators for phase
    sum_cos = 0.0
    sum_ang_deg = 0.0
    count_phase = 0.0

    for (x_in, y_true) in ds:
        y_s_true, y_e_true, y_p_true = y_true  # (B,64,800,1), (B,1,800,1), (B,64,800,2)
        y_s_pred, y_e_pred, y_p_pred = model(x_in, training=False)

        # ---- R² scalogram ----
        y_t = tf.reshape(tf.cast(y_s_true, tf.float32), [-1])
        y_p = tf.reshape(tf.cast(y_s_pred, tf.float32), [-1])
        err = y_t - y_p
        sse_scal  += float(tf.reduce_sum(tf.square(err)))
        sumy_scal += float(tf.reduce_sum(y_t))
        sumy2_scal+= float(tf.reduce_sum(tf.square(y_t)))
        n_scal    += int(tf.size(y_t))

        # ---- R² envelope ----
        y_t = tf.reshape(tf.cast(y_e_true, tf.float32), [-1])
        y_p = tf.reshape(tf.cast(y_e_pred, tf.float32), [-1])
        err = y_t - y_p
        sse_env   += float(tf.reduce_sum(tf.square(err)))
        sumy_env  += float(tf.reduce_sum(y_t))
        sumy2_env += float(tf.reduce_sum(tf.square(y_t)))
        n_env     += int(tf.size(y_t))

        # ---- Phase mean angle (deg) ----
        yt = tf.cast(y_p_true, tf.float32)
        yp = tf.cast(y_p_pred, tf.float32)
        yt = yt / (tf.norm(yt, axis=-1, keepdims=True) + eps)
        yp = yp / (tf.norm(yp, axis=-1, keepdims=True) + eps)
        cos_d = tf.reduce_sum(yt * yp, axis=-1)                      # (B,H,W)
        cos_d = tf.clip_by_value(cos_d, -1.0, 1.0)
        ang_deg = tf.acos(cos_d) * (180.0 / np.pi)

        sum_cos     += float(tf.reduce_sum(cos_d))
        sum_ang_deg += float(tf.reduce_sum(ang_deg))
        count_phase += float(tf.size(ang_deg))

    # Finalize R²
    sst_scal = sumy2_scal - (sumy_scal**2) / max(n_scal, 1e-12)
    sst_env  = sumy2_env  - (sumy_env**2)  / max(n_env,  1e-12)
    r2_scalo = 1.0 - sse_scal / max(sst_scal, 1e-12)
    r2_env   = 1.0 - sse_env  / max(sst_env,  1e-12)

    # Finalize phase metrics
    mean_cos       = sum_cos / max(count_phase, 1e-12)
    angular_loss   = 1.0 - mean_cos
    mean_angle_deg = sum_ang_deg / max(count_phase, 1e-12)

    return {
        "r2_scalogram": r2_scalo,
        "r2_envelope":  r2_env,
        "phase_mean_angle_deg": mean_angle_deg,
        "phase_mean_cos": mean_cos,
        "phase_angular_loss": angular_loss,
    }

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def check_dataset_for_nan_inf(dataset, dataset_name="Dataset"):
    for batch in dataset:
        if tf.reduce_any(tf.math.is_nan(batch)):
            print(f"Warning: NaN values found in {dataset_name}!")
        if tf.reduce_any(tf.math.is_inf(batch)):
            print(f"Warning: Inf values found in {dataset_name}!")
    print(f"{dataset_name} checked. No NaN or Inf found.")

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

# Shaping latent reps
def temporal_bin_pool(Z, n_bins: int):
    """Z: [N, T, D] -> [N, n_bins*D], mean over equal time bins."""
    N, T, D = Z.shape
    assert T % n_bins == 0, f"T={T} must be divisible by n_bins={n_bins}"
    L = T // n_bins
    Zb = Z.reshape(N, n_bins, L, D).mean(axis=2)   # [N, n_bins, D]
    return Zb.reshape(N, -1)                       # [N, n_bins*D]

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
    
def eval_latent_space(
    proj_model,                 # submodel that outputs the 150-D projection (e.g., "Proj_LN_1")
    data_inputs,                # [X_scalo_crop, X_env_crop, X_sin_crop, X_cos_crop]
    X_label,                    # your ground-truth phase labels (array-like, shape [N])
    dec_model=None,             # full DEC model (to set cluster_layer weights)
    temporal_bin=False,
    n_bins=10,
    n_clusters=3,
    cluster_ids=[1,0,2],
    title_1="Pre-Trained Latent Representations (True Labels)",
    title_2="Pre-Trained Latent Representations (Gaussian Mixture Labels)",
    title_3="Pre-Trained Confusion Matrix",
    gmm_kwargs=None,
    standardize=True,
    initialize_centroids=False,
    compute_acc=True,
    plot=False,
    cluster_layer_name="cluster_layer",
    random_state=42
):
    """
    1) Forward pass to get projection features (e.g., [N, 64]).
    2) Standardize (optional) and fit GMM in that space.
    3) Compute ARI/AMI/NMI (+ optional Hungarian-mapped accuracy).
    4) Initialize DEC cluster centroids from GMM means (inverse-transform if standardized).

    Returns a dict with metrics and fitted objects.
    """
    if gmm_kwargs is None:
        gmm_kwargs = dict(
            n_components=n_clusters,
            covariance_type="tied",
            n_init=3,
            init_params="kmeans",
            max_iter=500,
            tol=1e-3,
            reg_covar=1e-5,
            random_state=random_state,
        )

    # 1) Projected features
    Z_proj = proj_model.predict(data_inputs, verbose=1)  # [N, 64]
    if temporal_bin == True:
        # ---- reduce to 150 dims
        Z_proj = temporal_bin_pool(Z_proj, n_bins=n_bins) # [~30,000, 150]
    assert Z_proj.ndim == 2, f"Expected 2D features, got {Z_proj.shape}"

    # 2) Standardize for GMM
    scaler = None
    X_for_gmm = Z_proj
    if standardize:
        scaler = StandardScaler().fit(Z_proj)
        X_for_gmm = scaler.transform(Z_proj)

    # 3) Fit GMM + labels/probs
    gmm = GaussianMixture(**gmm_kwargs).fit(X_for_gmm)
    labels = gmm.predict(X_for_gmm)
    probs  = gmm.predict_proba(X_for_gmm)

    # 4) External clustering metrics (permutation-invariant)
    true_labels = get_true_labels(X_label, n_clusters=n_clusters, direct_id=0, trop_id=1, strat_id=2).astype(int)
    ari = adjusted_rand_score(true_labels, labels)
    ami = adjusted_mutual_info_score(true_labels, labels, average_method='arithmetic')
    nmi = normalized_mutual_info_score(true_labels, labels)

    # Silhouette in the same space used for clustering
    try:
        sil = silhouette_score(X_for_gmm, labels, metric='euclidean')
    except Exception:
        sil = np.nan

    # Optional: mapped accuracy via Hungarian
    mapped_acc, mapping = None, None
    if compute_acc:
        cm = confusion_matrix(true_labels, labels)
        row_ind, col_ind = linear_sum_assignment(-cm)
        mapping = {pred: true for pred, true in zip(col_ind, row_ind)}
        labels_aligned = np.array([mapping[l] for l in labels])
        mapped_acc = accuracy_score(true_labels, labels_aligned)

    counts = np.bincount(labels, minlength=n_clusters)
    print(f"GMM counts: {counts.tolist()}")
    print(f"ARI: {ari:.4f} | AMI: {ami:.4f} | NMI: {nmi:.4f} | Silhouette: {sil:.4f}")
    if mapped_acc is not None:
        print(f"Hungarian-mapped accuracy: {mapped_acc:.4f} | Mapping: {mapping}")

    # 5) Initialize DEC cluster centroids (GMM means back in raw projection space)
    if initialize_centroids == True and dec_model is not None:
        centroids = gmm.means_
        if standardize and scaler is not None:
            centroids = scaler.inverse_transform(centroids)  # back to the raw 300-D space that the layer sees
        cl = dec_model.get_layer(cluster_layer_name)
        cl.set_weights([centroids.astype(np.float32)])
        print(f"Initialized `{cluster_layer_name}` centroids with shape {centroids.shape}.")
    else:
        centroids = np.zeros((gmm.means_.shape))

    if plot:
        print("Plotting T-SNE")
        # --- t-SNE projection ---
        # Adjust perplexity based on your sample count:
        #  - 5–30 if N < 5000
        #  - 30–50 if N > 5000
        tsne = TSNE(
            n_components=2,
            perplexity=20,          # good default for 150-D latent space
            learning_rate='auto',
            init='pca',
            max_iter=1500,
            random_state=random_state,
            verbose=0
        )
        z_tsne = tsne.fit_transform(X_for_gmm)   # shape [N,150]
        
        # --- Plot ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
        
        # Panel 1: colored by true labels
        true_label_to_color = {0: 'black', 1: 'red', 2: 'green'}
        true_colors = np.array([true_label_to_color[label] for label in true_labels])
        axes[0].scatter(z_tsne[:, 0], z_tsne[:, 1], c=true_colors, s=1)
        axes[0].set_title(title_1)
        axes[0].set_xlabel("t-SNE 1")
        axes[0].set_ylabel("t-SNE 2")
        axes[0].set_xticks([]); axes[0].set_yticks([])
        for label, color in true_label_to_color.items():
            if color == "black": label_tmp = "Direct"
            if color == "red":   label_tmp = "Trop"
            if color == "green": label_tmp = "Strat"
            axes[0].scatter([], [], c=color, label=label_tmp, s=20)
        axes[0].legend(loc='lower left')

        # Panel 2: colored by cluster
        label_to_color = {cluster_ids[0]: 'black', cluster_ids[1]: 'red', cluster_ids[2]: 'green'}
        colors = np.array([label_to_color[label] for label in labels])
        axes[1].scatter(z_tsne[:, 0], z_tsne[:, 1], c=colors, s=1)
        axes[1].set_title(title_2)
        axes[1].set_xlabel("t-SNE 1")
        axes[1].set_xticks([]); axes[1].set_yticks([])
        for label, color in label_to_color.items():
            if color == "black": label_tmp = "Direct"
            if color == "red":   label_tmp = "Trop"
            if color == "green": label_tmp = "Strat"
            axes[1].scatter([], [], c=color, label=label_tmp, s=20)
        axes[1].legend(loc='lower left')
        plt.tight_layout()
        #-----------------------------------------------------------------------------------------------------------------------#
        print('Plotting PCA')
        # PCA 2 components
        z_pca = PCA(n_components=2).fit_transform(X_for_gmm)
        
        # --- Plot ---
        fig, axes = plt.subplots(1, 2, figsize=(14,6), sharex=True, sharey=True)
        
        # --- Panel 1: no labeling (all gray) ---
        axes[0].scatter(z_pca[:, 0], z_pca[:, 1], c=true_colors, s=1)
        axes[0].set_title(title_1)
        axes[0].set_xlabel("PCA Component 1")
        axes[0].set_ylabel("PCA Component 2")
        axes[0].set_xticks([]); axes[0].set_yticks([])
        for label, color in true_label_to_color.items():
            if color == "black": label_tmp = "Direct"
            if color == "red":   label_tmp = "Trop"
            if color == "green": label_tmp = "Strat"
            axes[0].scatter([], [], c=color, label=label_tmp, s=20)
        axes[0].legend(loc='lower left')
        
        # --- Panel 2: colored by cluster ---
        axes[1].scatter(z_pca[:, 0], z_pca[:, 1], c=colors, s=1)
        axes[1].set_title(title_2)
        axes[1].set_xlabel("PCA Component 1")
        axes[1].set_xticks([]); axes[1].set_yticks([])
        for label, color in label_to_color.items():
            if color == "black": label_tmp = "Direct"
            if color == "red": label_tmp = "Trop"
            if color == "green": label_tmp = "Strat"
            axes[1].scatter([], [], c=color, label=label_tmp, s=20)
        axes[1].legend(loc='lower left')
        plt.tight_layout()
        #-----------------------------------------------------------------------------------------------------------------------#
        # Confusion matrix
        cm_raw, cm_reordered, matched_cols = best_match_confusion(true_labels, labels)
        cm_percent = cm_reordered.astype(np.float32) / cm_reordered.sum(axis=1, keepdims=True) * 100
        fig, ax = plt.subplots(figsize=(8, 6))
        ticklabels = ["Direct", "Trop", "Strat"]
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Greys', cbar=True,
                    xticklabels=ticklabels,
                    yticklabels=ticklabels,
                    ax=ax)
        ax.set_title(title_3)
        ax.set_xlabel("Predicted Cluster")
        ax.set_ylabel("True Class")
        plt.tight_layout()

    return {
        "Z_proj": Z_proj,          # raw projected features [N, 64]
        "scaler": scaler,          # StandardScaler or None
        "gmm": gmm,                # fitted GMM
        "labels": labels,
        "probs": probs,
        "counts": counts,
        "ari": ari,
        "ami": ami,
        "nmi": nmi,
        "silhouette": sil,
        "mapped_acc": mapped_acc,
        "mapping": mapping,
        "centroids_loaded": centroids,  # centroids actually loaded to the layer (unstandardized)
    }
'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
# Scrap Code - revisit later
'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
# Time downsample and feature quantity specification for latent space
def proj_grid_stride(x, T_prime, K, name):
    T = x.shape[1]
    if (T is None) or (T % T_prime != 0):
        # fall back to Option A if not divisible
        return proj_grid_simple(x, T_prime, K, name)
    s = T // T_prime
    # kernel=5, stride=s downsamples and projects to K in one shot
    h = Conv1D(K, kernel_size=5, strides=s, padding="same",
               use_bias=True, kernel_initializer="he_uniform",
               name=f"{name}_conv_s{s}")(x)                                   # [B, T', K]
    return h

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

class AttentionPool1D(layers.Layer):
    def __init__(self, hidden=64, name_prefix="attnpool", **kwargs):
        super().__init__(**kwargs)
        self.dense_h = layers.Dense(hidden, name=f"{name_prefix}_h")
        self.dense_a = layers.Dense(1, name=f"{name_prefix}_a")

    def call(self, x):
        # x: [B, T, C]
        h = tf.nn.tanh(self.dense_h(x))           # [B, T, H]
        a = tf.nn.softmax(self.dense_a(h), axis=1) # [B, T, 1]
        return tf.reduce_sum(a * x, axis=1)       # [B, C]

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

# --- Projection head: 1x1 conv -> (optional) smoothing -> shared/private vectors ---
def projection_head_1d(x, Cp=64, d_shared=128, d_private=64, l2_lambda=1e-4, name_prefix="mod"):
    """
    x: [B, T, C]  (e.g., your tensor after the TransformerBlock)
    returns: (s_token [B, d_shared], p_token [B, d_private])
    """

    # 1x1 projection to Cp channels
    h = layers.Conv1D(
        Cp, kernel_size=1, padding='same',
        kernel_initializer='he_uniform',
        kernel_regularizer=regularizers.l2(l2_lambda),
        name=f"{name_prefix}_proj_1x1")(x)
    h = BatchNormalization()(h)
    h = activations.relu(h)
    
    # optional local smoothing with residual (can remove if you want it simpler)
    h_sm = layers.SeparableConv1D(
        Cp, kernel_size=3, padding='same',
        depthwise_regularizer=regularizers.l2(l2_lambda),
        pointwise_regularizer=regularizers.l2(l2_lambda),
        name=f"{name_prefix}_sep3")(h)
    h = layers.Add(name=f"{name_prefix}_res_add")([h, h_sm])  # [B, T, Cp]

    # pool over time to get summary vectors in ℝ^Cp
    s_pool = AttentionPool1D(hidden=max(32, Cp//2), name_prefix=f"{name_prefix}_poolS")(h)
    p_pool = AttentionPool1D(hidden=max(32, Cp//2), name_prefix=f"{name_prefix}_poolP")(h)

    # map to shared/private dims
    s_token = layers.Dense(d_shared, kernel_regularizer=regularizers.l2(l2_lambda),
                           name=f"{name_prefix}_shared")(s_pool)    # [B, d_shared]
    p_token = layers.Dense(d_private, kernel_regularizer=regularizers.l2(l2_lambda),
                           name=f"{name_prefix}_private")(p_pool)   # [B, d_private]
    return s_token, p_token
    
'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def compute_q(latent, centroids, alpha=1.0):
    """
    Computes soft assignments q using Student's t-distribution.

    Parameters:
    - latent: [N, D] latent vectors
    - centroids: [K, D] cluster centers

    Returns:
    - q: [N, K] soft assignments
    """
    # Compute pairwise squared distances between latent and centroids
    diff = latent[:, np.newaxis, :] - centroids[np.newaxis, :, :]  # shape: [N, K, D]
    dist_sq = np.sum(np.square(diff), axis=2)  # shape: [N, K]
    #-----------------------------------------------------------------------------------------------------------------------#
    # Student's t-distribution kernel
    numerator = (1.0 + dist_sq / alpha) ** (- (alpha + 1.0) / 2.0)
    q = numerator / np.sum(numerator, axis=1, keepdims=True)

    return q

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def build_DeepEmbeddedClustering_Model(X_cwt_scaled, X_envelope_scaled, X_sin_scaled, n_clusters, latent_space_size='regular'):

    infracoder, _ = build_InfraCoder(X_cwt_scaled, X_envelope_scaled, X_sin_scaled, lr=1e-2, l2_lambda=1e-4, return_attention=True, deep_embedding_clustering=True, return_attn_for_dec=True, latent_space_size=latent_space_size)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Construct latent model from InfraCoder
    latent_model = Model(inputs=infracoder.input, outputs=infracoder.get_layer("Merged_Latent_Space").output)
    latent_tensor = latent_model.get_layer('Merged_Latent_Space').output
    latent_flat = Flatten()(latent_tensor)
    #-----------------------------------------------------------------------------------------------------------------------#
    # Iniate clustering layer with centroids from kmeans
    clustering_layer = ClusteringLayer(n_clusters=n_clusters, latent_dim=latent_flat.shape[-1], name="Clustering_Layer") # define layer
    kl_loss = clustering_layer(latent_flat) # build layer
    #-----------------------------------------------------------------------------------------------------------------------#
    # Build DEC model
    scal_out, env_out, sin_out, cos_out, scal_attn, env_attn, phase_attn = infracoder.output # infracoder outputs including the attn scores
    outputs = [scal_out, env_out, sin_out, cos_out, kl_loss, scal_attn, env_attn, phase_attn]
    dec_model = Model(inputs=infracoder.input, outputs=outputs, name='Deep_Embedded_Clustering')

    return dec_model

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def get_true_labels(X_label, n_clusters=2, direct_id=0, refracted_id=1, trop_id=1, strat_id=2):

    gt_labels = []
    for instance in range(len(X_label)):
        gt_labels.append(X_label[instance][0].split('_')[1][:3])
    #-----------------------------------------------------------------------------------------------------------------------#    
    gt_cluster_labels = np.zeros((X_label.shape[0],))
    if n_clusters == 2:
        for idx, label in enumerate(gt_labels):
            if (label == 'MCA') or (label == 'MCB'):
                gt_cluster_labels[idx] = int(direct_id)
            else:
                gt_cluster_labels[idx] = int(refracted_id)
    elif n_clusters == 3:
        for idx, label in enumerate(gt_labels):
            if (label == 'MCA') or (label == 'MCB'):
                gt_cluster_labels[idx] = int(direct_id)
            elif label == 'EOC':
                gt_cluster_labels[idx] = int(trop_id)
            elif label == 'WCT':
                gt_cluster_labels[idx] = int(strat_id)

    return gt_cluster_labels

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def best_match_confusion(true_labels, pred_labels):

    cm = confusion_matrix(true_labels, pred_labels)
    _, col_ind = linear_sum_assignment(-cm)
    cm_reordered = cm[:, col_ind]
    return cm, cm_reordered, col_ind
