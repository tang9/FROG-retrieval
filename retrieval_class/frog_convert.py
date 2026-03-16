"""
Core FROG conversion utilities.

This module provides:
- `FROG`: loading, preprocessing, masking, binning, and exporting FROG traces.

Typical flow:
1. Read raw trace from file(s) or array-like input.
2. Apply optional background/noise processing.
3. Build binned FRG data of size `N x N`.
4. Export processed traces via `output_binned()`.
"""
#%%
import time
from os import PathLike
from pathlib import Path
from typing import Sequence
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.ndimage import gaussian_filter

from .common import find_index, bin3, plot_1d, plot_2d, fit_peak, gaussian_function

speed_light = 299792458.0
StrArrayLike = Sequence[str] | np.ndarray

def _fit_gaussian(x, y):
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    fit = SimpleNamespace(
        param=np.array([1.0, 0.0, 1.0], dtype=float),
        area=1.0,
        center=0.0,
        sigma=1.0,
        yfit=np.zeros_like(x_arr, dtype=float),
    )
    if x_arr.size > 3 and y_arr.size > 3:
        param, _ = fit_peak(x_arr, y_arr)
        fit.param = np.asarray(param, dtype=float)
        if fit.param.size == 3 and np.isfinite(fit.param[2]) and abs(fit.param[2]) > 0:
            fit.yfit = gaussian_function(x_arr, fit.param[0], fit.param[1], fit.param[2])
            fit.area = float(fit.param[0])
            fit.center = float(fit.param[1])
            fit.sigma = float(fit.param[2])
    return fit


# Main data pipeline for FROG trace conversion and FRG binning.
class FROG:
    def __init__(self, input:str|StrArrayLike|dict, wavelength_range=None, wavelength_bin = 1, \
                 noise_filter="", sigma=1, cutoff=6, filter_axis="wavelength", \
                 delay_step="mm", delay_correction=2.0,\
                 edge = 10, subx=True, suby=True, \
                 corner_suppression = False, mask_frg = None, constant_bkg = 0,\
                 time_zero=True, time_range = None, raw=False, N = 128, profile=False):
        """
        Parameters
        ----------
        input : str or list of str or dict{delay, wavelength,frog_trace,prefix}.
            filename of the frog trace. If the input is a list of filenames, it will average them.
            input file always have the format:
            line1: delays
            line2: wavelengths
            Next: FROG TRACE MATRIX, each row is the spectrum at each delay
        wavelength_range : tuple. the range of wavelength that will be used
        wavelength_bin : int. the binning step size in the wavelength axis, to reduce noise
        noise_filter : str. Can be "" (no noise filter),"Gaussian" (Gaussian filter, default sigma=1) or "Fourier" (Fourier low pass filter).
        cutoff : int. Only used for Fourier low pass filter, to determine the frequency cut-off of the low pass filter
        filter_axis : str ["wavelength", "delay", or "both"]. Only used for Fourier low pass filter, determine which axis will be filters
        delay_step : str ["m","mm","um","nm","ps","fs","as"]. the unit of the delays in the first line of the data
        delay_correction : int. default 2.0. Normally the delay is twice of the delay stage. 
            Sometimes the actuator does not move as it shows in the program. We need to check that every time after frogging.
        edge : int. the pixel range at the edge of the matrx that treated as noise level. The default is 20.
        subx : True or Flase. Whether to subtract backgroud according to the edge data in delay axis.
        suby : True or Flase. Whether to subtract backgroud according to the edge data in wavelength axis.
        corner_suppression: using super-Gaussian function to suppress the noise at the edge (20% region)
        mask_trace : True or Flase. If true, using a super-Gaussian function to suppress the corner part, 
                    and mask the data whose absolute value is less than certian value (max of 1 and edge max value) to be zero.
        mask_frg : float. The threshold below which the data in binned frg will be set as zero. It will do corner suppression first.
        time_zero : True or False. Shift the delay to make the center as zero delay.
        time_range : tuple, limit or expand the time range.
        raw : True or False. If it is true, only read the data, no further processing
        N : int [64, 128, 256, 512, 1024]. The width and height of the binned data.
        profile : True or False. If true, print step-by-step timing to stdout.
        ---------
        """
        self.profile_enabled = bool(profile)
        self.profile_timing = []
        self.profile_total = 0.0
        t_start = time.perf_counter()
        t_last = t_start

        def _mark(step_name: str):
            nonlocal t_last
            if not self.profile_enabled:
                return
            t_now = time.perf_counter()
            self.profile_timing.append((step_name, t_now - t_last))
            t_last = t_now

        def _finish(stage_name: str):
            if not self.profile_enabled:
                return
            self.profile_total = time.perf_counter() - t_start
            summary = " | ".join([f"{k}={v:.3f}s" for k, v in self.profile_timing])
            #print(f"[FROG timing] {stage_name} total={self.profile_total:.3f}s | {summary}")

        try:
            r = self.read_frog(input,wavelength_range,wavelength_bin, delay_step,delay_correction)
            _mark("read_frog")
            if (not r) or len(self.delay) < 10:
                _finish("early_return")
                return
            #### substract backgroud
            self.subtract_bkg(edge,subx,suby)
            _mark("subtract_bkg")
            #set the timezero according to the fitting of the autocorrelation
            self.autocorrelation = np.sum(self.frog_trace,axis = 1)
            self.autocorrelation_fit = _fit_gaussian(self.delay, self.autocorrelation)
            _mark("init_autocorrelation")
            if time_zero:
                self.set_timezero()
                _mark("set_timezero")
            if time_range:
                self.crop_time(time_range)
                _mark("crop_time")
            if raw:
                _finish("raw_only")
                return
            #### Noise Filter
            if noise_filter.find("Gaussian") != -1:
                self.GaussianFilter(sigma,axis=filter_axis or "both")
            elif noise_filter.find("Fourier") != -1:
                self.FourierFilter("Butterworth",cutoff,axis=filter_axis or "both")
            _mark(f"noise_filter:{noise_filter or 'none'}")
            #resudce constant background
            self.frog_trace = self.frog_trace - constant_bkg*np.max(self.frog_trace)
            self.autocorrelation = np.sum(self.frog_trace,axis = 1)
            self.autocorrelation_fit = _fit_gaussian(self.delay, self.autocorrelation)
            # keep a copy before normalization for GUI 1D comparison
            self.frog_trace_pre_norm = np.array(self.frog_trace, copy=True)
            self.autocorrelation_pre_norm = np.array(self.autocorrelation, copy=True)
            # normalize the autocorrelation to the max value
            max_trace = np.max(self.frog_trace)
            if max_trace > 0:
                self.frog_trace /= max_trace
            max_auto = np.max(self.autocorrelation)
            if max_auto > 0:
                self.autocorrelation /= max_auto
            _mark("normalize")

            if corner_suppression:
                self.frog_trace = self.corner_suppression(self.frog_trace)
                self.autocorrelation = np.sum(self.frog_trace,axis = 1)
                self.autocorrelation_fit = _fit_gaussian(self.delay, self.autocorrelation)
                _mark("corner_suppression")

            #### binner to N*N time-frequency frog trace
            self.frg_bin_size = N or 256
            self.freq = speed_light/self.wavelength/1E-9/1E15
            self.generate_binned_trace(self.frg_bin_size)
            _mark("generate_binned_trace")
            
            # if corner_suppression:
            #     self.frg = self.corner_suppression_frg(self.frg, self.frg_bin_size)
            #     _mark("corner_suppression")
            
            ###mask the data
            if mask_frg is not None and self.frg is not None:
                G = self.frg > mask_frg
                self.frg *= G
                _mark("mask_frg")
            _finish("processed")
        except Exception as e:
            _finish("failed")
            print(f"Error processing FROG data: {e}")     
            
            
    def read_frog(self,input,wavelength_range,wavelength_bin,delay_step,delay_correction):
        # fread frog data into frog_trace
        if not input:
            print("No input file!")
            return False
        if delay_correction == 0:
            print("Invalid delay_correction: 0")
            return False
        if isinstance(input, dict) and "wavelength" in input and "delay" in input and "frog_trace" in input:
            self.wavelength = np.array(input["wavelength"])
            self.delay = np.array(input["delay"])
            self.frog_trace = np.array(input["frog_trace"])
            self.prefix = input.get("prefix", "input_trace")
        else:
            if isinstance(input, (str, PathLike)):
                filenames = [str(input)]
            elif isinstance(input, (list, tuple, np.ndarray)):
                try:
                    items = list(input)
                except TypeError:
                    print("Invalid input sequence type for filenames.")
                    return False
                if not all(isinstance(t, (str, PathLike)) for t in items):
                    print("Invalid filename list: all elements must be str or PathLike.")
                    return False
                filenames = [str(t) for t in items]
                if len(filenames) == 0:
                    print("No input file!")
                    return False
            else:
                print("Invalid input type! Use filename, list of filenames, or dict with keys 'wavelength', 'delay', and 'frog_trace'.")
                return False

            self.delay, self.wavelength = [], []
            each_trace, self.frog_trace = [], []
            self.prefix = self.remove_extension(filenames[0])
            if len(filenames) > 1:
                self.prefix += "_average"
            factor_dict = {"nm": 1e-6, "mm": 1.0, "um": 1e-3, "m": 1000, \
                           "fs": 1e-15 * speed_light * 1e3, "ps": 1e-12 * speed_light * 1e3, "as": 1e-18 * speed_light * 1e3}
            try:
                delay_scale = factor_dict[str(delay_step).lower()] # convert input delay step to motor position in mm
            except KeyError:
                print(f"Invalid delay_step: {delay_step}. Use 'nm', 'mm', 'um', 'm', 'fs', 'ps', or 'as'.")
                return False
            ref_delay, ref_wavelength = None, None
            for i_file, file in enumerate(filenames):
                try:
                    with open(file, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                except Exception as e:
                    print(f"Error reading file {file}: {e}")
                    return False
                # Skip title/comments until first numeric row.
                start_idx = None
                for i_line, line in enumerate(lines):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        [float(x) for x in line.split()]
                        start_idx = i_line
                        break
                    except Exception:
                        continue
                if start_idx is None or (len(lines) - start_idx) < 2:
                    print(f"Invalid file format: {file}")
                    return False
                lines = lines[start_idx:]
                try:
                    motor = [float(x) * delay_scale for x in lines[0].split()]
                    wavelength = [float(x) for x in lines[1].split()]
                except ValueError:
                    print(f"Invalid numeric data in header: {file}")
                    return False
                if len(motor) == 0 or len(wavelength) == 0:
                    print(f"Empty delay/wavelength data: {file}")
                    return False
                # Convert motor positions to delay in fs, considering the correction factor.
                delay = [m * 1e-3 / speed_light * 1e15 * delay_correction for m in motor]

                if i_file == 0:
                    ref_delay = delay
                    ref_wavelength = wavelength
                    self.delay = np.array(delay)
                    self.wavelength = np.array(wavelength)
                else:
                    if len(ref_delay) != len(delay) or any(np.abs(a - b) > 0.1 for a, b in zip(ref_delay, delay)):
                        print("delay wrong for files to average!")
                        return False
                    if len(ref_wavelength) != len(wavelength) or any(np.abs(a - b) > 0.01 for a, b in zip(ref_wavelength, wavelength)):
                        print("wavelength wrong for files to average!")
                        return False
                rows = []
                for line in lines[2:]:
                    t = line.split()
                    if t:
                        try:
                            rows.append([float(x) for x in t])
                        except ValueError:
                            print(f"Invalid matrix row in file: {file}")
                            return False
                each_trace.append(np.array(rows))
            self.frog_trace = np.mean(each_trace, axis=0)  # [delay,wavelength]
            
        self.motor = self.delay * 1e-15 * speed_light * 1e3 / delay_correction  # convert back to motor position in mm
        # Check the trace orientation. Prefer the documented file format:
        # rows = delay, cols = wavelength. Only transpose when that is
        # unambiguously inconsistent with the header axes.
        shape = self.frog_trace.shape
        if shape[0] == len(self.delay) and shape[1] == len(self.wavelength):
            pass
        elif shape[1] == len(self.delay) and shape[0] == len(self.wavelength):
            self.frog_trace = self.frog_trace.T
            shape = self.frog_trace.shape
        if shape[0] != len(self.delay) or shape[1] != len(self.wavelength):
            print("Warning: delay or wavelength point number wrong!")
            return False

        print("Frog trace raw data: delay %d: %f~%f fs, wavelength %d: %f~%f nm"\
              % (shape[0], self.delay[0], self.delay[-1], shape[1], self.wavelength[0], self.wavelength[-1]))
        # set the wavelength range
        if wavelength_range:
            w1 = find_index(self.wavelength, wavelength_range[0])
            w2 = find_index(self.wavelength, wavelength_range[1])
            self.wavelength = self.wavelength[w1:w2]
            self.frog_trace = self.frog_trace[:, w1:w2]

        # remove duplicated delay
        count_dup = 0
        i = 0
        while i < len(self.delay) - 1:
            if (self.delay[i + 1] - self.delay[i]) < 0.1:  # less 0.1 fs, delete
                self.delay = np.delete(self.delay, i)
                self.frog_trace = np.delete(self.frog_trace, i, axis=0)
                count_dup += 1
            else:
                i += 1
        if count_dup > 0:
            print("%d duplicated delay" % count_dup)

        # binning in the wavelength axis
        if wavelength_bin > 1:
            if wavelength_bin > len(self.wavelength):
                print(f"wavelength_bin={wavelength_bin} is larger than wavelength size {len(self.wavelength)}")
                return False
            self.delay,self.wavelength,self.frog_trace = bin3(self.delay,self.wavelength,self.frog_trace,wavelength_bin,axis=1,result="average")
        if self.frog_trace.size == 0:
            print("Empty frog_trace after processing")
            return False
        shape = self.frog_trace.shape
        print("Frog trace read data: delay %d: %f~%f fs, wavelength %d: %f~%f nm"\
              % (shape[0], self.delay[0], self.delay[-1], shape[1], self.wavelength[0], self.wavelength[-1]))
        return True

    def remove_extension(self,s):
        ss = s.split(".")
        return s[:len(s)-len(ss[-1])-1]
    
    def subtract_bkg(self,edge=20,subx=True,suby=True):
        # Subtract linear background estimated from both edges.
        if self.frog_trace is None or np.size(self.frog_trace) == 0:
            return
        edge = int(edge)
        if edge < 1:
            edge = 1
        edge = min(edge, len(self.delay), len(self.wavelength))

        if subx:
            # For each wavelength column, fit a linear background vs delay
            # using points from [:edge] and [-edge:].
            x = np.asarray(self.delay, dtype=float)
            idx = np.r_[0:edge, len(x)-edge:len(x)]
            x_fit = x[idx]
            for j in range(len(self.wavelength)):
                y_fit = self.frog_trace[idx, j]
                m, b = np.polyfit(x_fit, y_fit, 1)
                self.frog_trace[:, j] -= (m * x + b)
        if suby:
            # For each delay row, fit a linear background vs wavelength
            # using points from [:edge] and [-edge:].
            x = np.asarray(self.wavelength, dtype=float)
            idx = np.r_[0:edge, len(x)-edge:len(x)]
            x_fit = x[idx]
            for i in range(len(self.delay)):
                y_fit = self.frog_trace[i, idx]
                m, b = np.polyfit(x_fit, y_fit, 1)
                self.frog_trace[i, :] -= (m * x + b)

    def GaussianFilter(self,sigma,axis="both"):
        axis_key = str(axis).strip().lower()
        if axis_key in ["d", "delay", "x"]:
            # delay axis only
            self.frog_trace = gaussian_filter(self.frog_trace, sigma=(sigma, 0))
        elif axis_key in ["w", "wavelength", "y"]:
            # wavelength axis only
            self.frog_trace = gaussian_filter(self.frog_trace, sigma=(0, sigma))
        elif axis_key in ["b", "both"]:
            self.frog_trace = gaussian_filter(self.frog_trace, sigma=sigma)
        else:
            print("GaussianFilter axis error! Use 'delay'/'x', 'wavelength'/'y', or 'both'.")
   
    def FourierFilter(self,filter_type="Butterworth",cutoff=6,axis="both"): 
        #do Fourier low pass filter to the frog trace
        axis_key = str(axis).strip().lower()
        trace = self.frog_trace
        #we want to get odd number of rows and columns
        delayadd,waveadd = 0,0
        if len(self.delay)%2 == 0:
            trace = trace[:len(self.delay)-1,:]
            delayadd = 1
        if len(self.wavelength)%2 == 0:
            trace = trace[:,:len(self.wavelength)-1]
            waveadd = 1
        """
        The values in the result follow so-called standard order: If A = fft(a, n), 
        then A[0] contains the zero-frequency term (the sum of the signal), which is 
        always purely real for real inputs. Then A[1:n/2] contains the positive-
        frequency terms, and A[n/2+1:] contains the negative-frequency terms, in order 
        of decreasingly negative frequency. For an even number of input points, A[n/2] 
        represents both positive and negative Nyquist frequency, and is also purely real 
        for real input. For an odd number of input points, A[(n-1)/2] contains the largest
        positive frequency, while A[(n+1)/2] contains the largest negative frequency. 
        The routine np.fft.fftfreq(n) returns an array giving the frequencies of 
        corresponding elements in the output. The routine np.fft.fftshift(A) shifts 
        transforms and their frequencies to put the zero-frequency components in the 
        middle, and np.fft.ifftshift(A) undoes that shift.
        """
        frog_fft = np.fft.fft2(trace)
        m,n = frog_fft.shape #ndelay, nwave, should be odd
        
        #determine the cut-off frequency of the low pass filter
        fit1 = _fit_gaussian(np.arange(m), np.sum(trace, axis=1))
        D1 = 0.1592*m/fit1.sigma*cutoff#6sigma
        #0.1592 = 1/2pi, frequency->round frequency
        #print("sigma_delay=%.2f, cut-off: %.2f"%(fit1.sigma,D1))

        fit2 = _fit_gaussian(np.arange(n), np.sum(trace, axis=0))
        D2 = 0.1592*n/fit2.sigma*cutoff#6sigma
        #print("sigma_wave=%.2f, cut-off: %.2f"%(fit2.sigma,D2))
        
        frog_fft = np.fft.fftshift(frog_fft) #shift the frequency, the center is low frequency
        #Center: m//2, n//2, 5: 2, 0-1&3-4
        #An ideal low-pass filter results in ringing artifacts via the Gibbs phenomenon
        #H[m//2-D1:m//2+D1+1,n//2-D2:n//2+D2+1]=1
        #filter_types=["Ideal","Gaussian","Butterworth","Hamming"]
        I, J = np.meshgrid(np.arange(m), np.arange(n), indexing='ij')
        if axis_key in ["d", "delay", "x"]:
            k = (I - m//2)**2 / D1**2
        elif axis_key in ["w", "wavelength", "y"]:
            k = (J - n//2)**2 / D2**2
        else:
            k = (I - m//2)**2 / D1**2 + (J - n//2)**2 / D2**2
        if filter_type == "Ideal":
            H = (k <= 1).astype(float)
        elif filter_type == "Butterworth":
            H = 1 / (1 + k**2)
        elif filter_type == "Gaussian":
            H = np.exp(-0.5 * k)
        elif filter_type == "Hamming":
            H = np.where(k <= 4, 0.5 * (1 + np.cos(np.pi * np.sqrt(k / 4))), 0.0)
        else:
            H = np.full((m, n), 1.0)
        frog_fft = frog_fft*H
        frog_fft = np.fft.ifftshift(frog_fft)
        trace = np.fft.ifft2(frog_fft).real
        self.frog_trace[:len(self.delay)-delayadd,:len(self.wavelength)-waveadd]=trace
    
    def corner_suppression(self,trace,k=8):
        #suppress the corner data (~20% range)
        m,n = trace.shape
        C1, C2 = m/2, n/2
        D1, D2 = m*0.4, n*0.4
        #super-Gaussian 
        X, Y = np.meshgrid(np.arange(m),np.arange(n))
        corner = np.exp( -((X-C1)**2/D1**2+(Y-C2)**2/D2**2)**k/2).T
        return trace * corner

    def corner_suppression_frg(self,trace,N,k=8):
        #suppress the corner data (~20% range) in the binned data
        m,n = trace.shape
        C1, C2 = N/2, N/2
        D1 = N*0.4
        D2 = N*0.4
        #super-Gaussian
        X, Y = np.meshgrid(np.arange(m),np.arange(n))
        corner = np.exp( -((X-C1)**2/D1**2+(Y-C2)**2/D2**2)**k/2).T
        return trace * corner
    
    def set_timezero(self):
        #revise time-zero        
        self.autocorrelation_fit = _fit_gaussian(self.delay, self.autocorrelation)
        center = self.autocorrelation_fit.center
        self.delay -= center
        self.autocorrelation_fit.param[1] = 0            
        print("Center: %.3f,Set_timezero: %d delays from %.2f fs to %.2f fs."%(center,len(self.delay),self.delay[0],self.delay[-1]))       
    
    def crop_time(self,time_range):
        #crop or expand the time range
        if len(self.delay) < 2:
            print("crop_time error: insufficient delay points.")
            return
        if time_range[0] >= time_range[1]:
            print("crop_time error: invalid time_range order.")
            return
        delay_step = (self.delay[-1]-self.delay[0])/(len(self.delay)-1)
        ind1 = find_index(self.delay,time_range[0])
        ind2 = find_index(self.delay,time_range[1])
        self.delay = self.delay[ind1:ind2]
        self.frog_trace = self.frog_trace[ind1:ind2,:]  #[delay,wavelength]
        if len(self.delay) < 2:
            print("crop_time error: insufficient delay range after slicing.")
            return
        wave_size = self.frog_trace.shape[1]
        if time_range[0] < self.delay[0] - delay_step: #expand the trace
            add_delay = np.arange(self.delay[0] - delay_step, time_range[0] - delay_step, -delay_step)[::-1]
            add_trace = np.zeros((len(add_delay),wave_size))
            self.delay = np.concatenate((add_delay,self.delay))
            self.frog_trace = np.concatenate((add_trace,self.frog_trace),axis=0)
        if time_range[1] > self.delay[-1] + delay_step: #expand the trace
            add_delay = np.arange(self.delay[-1] + delay_step, time_range[1] + delay_step, delay_step)
            add_trace = np.zeros((len(add_delay),wave_size))
            self.delay = np.concatenate((self.delay,add_delay))
            self.frog_trace = np.concatenate((self.frog_trace,add_trace),axis=0)
        shape = self.frog_trace.shape
        print("Adjusting time range: delay %d: %f~%f fs, wavelength %d: %f~%f nm"\
              %(shape[0],self.delay[0],self.delay[-1],shape[1],self.wavelength[0],self.wavelength[-1]))
        self.autocorrelation = np.sum(self.frog_trace,axis = 1)
        self.autocorrelation_fit = _fit_gaussian(self.delay, self.autocorrelation)

    def generate_binned_trace(self,N:int):
        if N not in [64,128,256,512,1024,2048]:
            print("Invalid N for binned trace! Use 64,128,256,512, 1024, or 2048.")
            return False
        if len(self.delay) < 2 or len(self.wavelength) < 2:
            print("generate_binned_trace error: insufficient delay/wavelength points.")
            return False
        self.sumspectrum = np.sum(self.frog_trace,axis = 0)
        self.freq = speed_light/self.wavelength/1E-9/1E15
        try:
            fit = _fit_gaussian(self.wavelength, self.sumspectrum)
            wave_center = fit.center
        except Exception:
            wave_center = np.mean(self.wavelength)
        if wave_center == 0:
            print("generate_binned_trace error: invalid wave center.")
            return False
        self.v0 = speed_light/wave_center*1e9/1e15 #center frequency in pHz

        delay_max = max(abs(self.delay[0]), abs(self.delay[-1]))
        self.dt = 2*delay_max/N
        if self.dt == 0:
            print("generate_binned_trace error: zero delay span.")
            return False
        self.dv = 1/(N*self.dt)
        x = (np.arange(N) - N // 2) * self.dt
        y = self.v0 + (np.arange(N) - N // 2) * self.dv

        delay_axis = np.asarray(self.delay, dtype=float).ravel()
        freq_axis = np.asarray(self.freq, dtype=float).ravel()
        wave_axis = np.asarray(self.wavelength, dtype=float).ravel()
        trace = np.asarray(self.frog_trace, dtype=float)
        if trace.shape == (delay_axis.size, wave_axis.size):
            pass
        elif trace.shape == (wave_axis.size, delay_axis.size):
            trace = trace.T
        else:
            print(
                f"generate_binned_trace error: shape mismatch trace={trace.shape}, "
                f"delay={delay_axis.size}, wavelength={wave_axis.size}"
            )
            return False

        # Convert spectral marginal from wavelength to frequency domain.
        values = trace * (wave_axis[None, :] / wave_center) ** 2

        # Ensure interpolation axes are monotonic increasing and unique.
        d_order = np.argsort(delay_axis)
        f_order = np.argsort(freq_axis)
        d_sorted = delay_axis[d_order]
        f_sorted = freq_axis[f_order]
        v_sorted = values[np.ix_(d_order, f_order)]

        d_unique, d_idx = np.unique(d_sorted, return_index=True)
        f_unique, f_idx = np.unique(f_sorted, return_index=True)
        v_unique = v_sorted[np.ix_(d_idx, f_idx)]
        if d_unique.size < 2 or f_unique.size < 2:
            print("generate_binned_trace error: insufficient unique delay/frequency points.")
            return False

        interpolator = RegularGridInterpolator(
            (d_unique, f_unique),
            v_unique,
            method="cubic",
            bounds_error=False,
            fill_value=0.0,
        )

        xq, yq = np.meshgrid(x, y, indexing="ij")
        query = np.column_stack((xq.ravel(), yq.ravel()))
        self.frg = interpolator(query).reshape(xq.shape)

        max_frg = np.max(self.frg)
        if max_frg > 0:
            self.frg /= max_frg
        else:
            print("generate_binned_trace warning: max(frg)==0, skip normalization.")
        self.delay_binned = x
        self.freq_binned = y
        return True
        
    def draw_frog_trace(self,ax=None,scale="linear",levels=20,colorbar=True,labelsize=12,\
        xlim=None,ylim=None,title=None,colorbar_ticks=None,shrink=1):
        plot_2d(ax,self.delay,self.wavelength,self.frog_trace,logscale=(scale=="log"),levels=levels,\
                colorbar=colorbar,colorbar_ticks=colorbar_ticks,shrink=shrink,\
                xlim=xlim,ylim=ylim,title=title,xlabel="Delay (fs)",ylabel="Wavelength (nm)",\
                labelsize=labelsize, stype="pcolormesh")
    
    def draw_binned_frg(self,ax=None,scale="linear",levels=20,addfigurebar=True,\
        xlim=None,ylim=None,title=None,labelsize=12):
        plot_2d(ax,self.delay_binned,self.freq_binned,self.frg,logscale=(scale=="log"),levels=levels,\
                colorbar=addfigurebar,xlim=xlim,ylim=ylim,title=title,xlabel="Delay (fs)",
                ylabel="Frequency (PHz)",labelsize=labelsize, stype="pcolormesh")

    def draw_autocorrelation_wave(self,ax=None,scale="linear",add_fit=True,\
                    xlim=None,ylim=None,title=None):
        ax = plot_1d(ax,self.delay,self.autocorrelation,linestyle='solid',linewidth=1,\
                marker=".",markersize=6,label="AutoCorrelation",yscale=scale,\
                xlim=xlim,ylim=ylim or (0,1.5),title=title,xlabel="Delay (fs)",ylabel="Intensity")
        if add_fit:
            ax.plot(self.delay,self.autocorrelation_fit.yfit/max(self.autocorrelation),linestyle='solid',linewidth=1,\
                label="Gaussian Fit, FWHM=%.1f"%(self.autocorrelation_fit.param[2]*2.35482))
        ax.legend()

    """Deprecated output method for raw-grid .frg export, no longer used by current GUI pipeline.
    def output_frg(self):
        #Legacy implementation kept here intentionally, but masked out.
        # === file structure of .frg, for matlab binned input ===
        # width(temporal domain) of FROG trace in pixel \t  
        # height(spectral domain) of FROG trace in pixel \t 
        # temporal_calibration (fs/px) \t  
        # spectral_calibration (nm/px) \t   
        # central_wavelength (nm) \t\n
        # FROG_TRACE_RAW_DATA
        # === EOF ===
        # Example 
        # === xxx.frg ===
        # 512	512	20.54542	9.506378e-05	0.3747406
        # [512 x 512] matrix
        # === EOF === 
        x,y,res = self.delay,self.wavelength,self.frog_trace
        width = len(x)
        height = len(y)
        t_calib = x[1]-x[0]  #fs/pixel
        wave_calib = y[1]-y[0] #nm/pixel
        wave_central = (y[0]+y[-1])/2
        prefix = self.prefix
        out_path = Path(prefix + ".frg")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path,'w') as f:
            f.write("%f\t%f\t%f\t%f\t%f\t\n"%(width,height,t_calib,wave_calib,wave_central))
            for j in range(len(y)):
                for i in range(len(x)):
                    f.write("%f\t"%res[i,j])
                f.write("\n")
    """

    def output_binned(self):
        """
        === file structure of _binned.frg ===
        width(N) \t height(N) \t temporal_calibration (fs/px) \t 
        frequency_calibration (PHz/px) \t  central_frequency (PHz) \t\n
        FROG_TRACE_MATRIX
        === EOF ===
        """
        x,y,res = self.delay_binned,self.freq_binned,self.frg
        N = len(x)
        prefix = self.prefix
        out_path = Path(prefix + "_binned%d.frg"%N)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path,'w') as f:
            f.write("%d\t%d\t%f\t%f\t%f\t\n"%(N,N,self.dt,self.dv,self.v0))
            for j in range(len(y)):
                for i in range(len(x)):
                    f.write("%f\t"%res[i,j])
                f.write("\n")

    def output_trace(self):
        """
        output the trackled data for slava's program
        firstline: delay(fs)
        second line: wavelength(nm)
        Remaining: spectra
        """
        prefix = self.prefix
        out_path = Path(prefix + "_processed.txtSpecScan")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path,'w') as f:
            #first line:
            for i,dly in enumerate(self.delay):
                f.write("%f"%dly)
                if i!= len(self.delay)-1:
                        f.write("\t")
            f.write("\n")
            #second line
            for j,wl in enumerate(self.wavelength):
                f.write("%f"%wl)
                if j!= len(self.wavelength)-1:
                        f.write("\t")
            f.write("\n")
            #remaining lines
            for i in range(len(self.delay)):
                for j in range(len(self.wavelength)):
                    f.write("%.1f"%self.frog_trace[i][j])
                    if j!= len(self.wavelength)-1:
                        f.write("\t")
                f.write("\n")
            f.close()
            
    def convolution(self,filename,center=2800,width=500):
        wave, intensity = [],[]
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[1:]:
                t = line.split()
                wave.append(float(t[0]))
                intensity.append(float(t[1]))
        i_min = find_index(wave,center-width)
        i_max = find_index(wave,center+width)
        wave,intensity = wave[i_min:i_max],intensity[i_min:i_max]
        freq = [speed_light/wave_i/1E-9/1E15 for wave_i in wave] #PHz=1000THz
        #f(w) = lambda^2/2/pi/speed_light*f(lambda)
        intens_freq = [wave[i]**2/wave[0]**2*intensity[i] for i in range(len(wave))]
        #fconv(w') = Integrate(f(w)*f(w'-w))
        #MSHG = I(w)(*)I(w)
        x = np.linspace(freq[0], freq[-1], num=len(freq), endpoint=True)
        fx = interp1d(freq, intens_freq, kind='cubic')
        y = [fx(xi) for xi in x]
        #(a*v)[n]=SIGMA_m(a[m]v[n-m])
        yy = np.convolve(y,y)
        xx = [2*x[0]+(x[1]-x[0])*i for i in range(len(yy))] #frequency
        #freq=speed_light/wave_i/1E-9/1E15, wave_i=speed_light/freq/1E-9/1E15
        x_wave = [speed_light/freq_i/1E-9/1E15 for freq_i in xx]
        #f(lambda) = w^2/2/pi/speed_light*f(w)
        autoconv  = np.array([yy[i]*xx[i]**2 for i in range(len(xx))])
        autoconv /= np.max(autoconv)
        #autoconv = [0]*len(self.wavelength)
        #self.freq = [speed_light/wave_i/1E-9/1E15 for wave_i in self.wavelength]
        #fx = interp1d(freq, intens_freq, kind='cubic')
        return x_wave,autoconv
    
    def check_freqMarginal(self,fundamental_filename,ax=None,center=2800,width=500):
        self.sumspectrum = np.sum(self.frog_trace,axis = 0)
        self.sumspectrum /= np.max(self.sumspectrum)
        w_conv,autoconv = self.convolution(fundamental_filename,center=center,width=width)
        if ax is None:
            fig = plt.figure(figsize=(8,6))
            ax = fig.add_subplot(111)
        ax.plot(self.wavelength,self.sumspectrum,".-",\
                label="%.1fum_"%(center/1000)+"FROG Marginal")
        ax.plot(w_conv,autoconv,label="%.1fum_"%(center/1000)+"Autoconvolution")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity")
        ax.legend(fontsize=12)
        ax.grid()
        plt.subplots_adjust(left=0.1,bottom=0.1,right=0.95,top=0.95,wspace=0.25,hspace=0.25)
        
        with open("check_marginals_FROG_marginals.txt", 'w', encoding='utf-8') as f:
            for i in range(len(self.wavelength)):
                f.write("%f\t%f\n"%(self.wavelength[i],self.sumspectrum[i]))
            f.close()
        with open("check_marginals_autoConvolution.txt", 'w', encoding='utf-8') as f:
            for i in range(len(w_conv)):
                f.write("%f\t%f\n"%(w_conv[i],autoconv[i]))
            f.close()
    
    def get_spectra(self):
        #self.freq_binned, self.v0, self.dv
        t0 = self.freq_binned[0] - self.v0
        t = np.arange(0,len(self.freq_binned))/(self.freq_binned[-1]-self.freq_binned[0])
        extra = np.exp(-2*np.pi*1.0j*t0*t)
        Mw = np.sum(self.frg,axis = 0)
        fft = np.fft.fft(Mw)*extra
        a = np.fft.ifft(np.sqrt(fft))
        return self.freq+self.v0, a


#%%super-Gaussian test
"""
N = 128
x = np.arange(0,N,1)
center = N/2
sigma = 8
y = np.exp(-((x-center)**2/sigma**2)/2)
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
ax.plot(x,y)
M = N*0.4/8
for k in [4,6,8]:
    y1 = y = np.exp(-((x-center)**2/(sigma*M)**2)**k/2)
    ax.plot(x,y1,".-",label="k=%d"%k)
ax.legend()
"""

#%% corner suppression test
def test_corner_suppression():
    def corner_suppression(trace,k=8):
        #suppress the corner data (~20% range)
        m,n = trace.shape
        C1, C2 = m/2, n/2
        D1, D2 = m*0.4, n*0.4
        #super-Gaussian 
        X, Y = np.meshgrid(np.arange(m),np.arange(n))
        corner = np.exp( -((X-C1)**2/D1**2+(Y-C2)**2/D2**2)**k/2).T
        return trace * corner
    trace = np.full((100,100),1.0)
    trace_cs = corner_suppression(trace)
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111)
    plot_2d(ax,np.arange(100),np.arange(100),trace_cs,logscale=False,levels=20,colorbar=True,\
            xlim=(0,100),ylim=(0,100),title="Corner Suppression",labelsize=12, style="pcolormesh")
    ax.grid()
    plt.tight_layout()
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111)
    plot_1d(ax,np.arange(100),[trace_cs[50,:]],fmt=['-'],\
            xlim=(0,20),ylim=(0,1.5),title="Corner Suppression")
    ax.grid()
    plt.tight_layout()
    plt.show()
#test_corner_suppression()



# %%
