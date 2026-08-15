from os import PathLike

import numpy as np

from .common import find_index, find_peak_fwhm, find_peak_ratio, fit_peak, get_Et_from_Ef

speed_light = 299792458.0

class FROG_result:
    def __init__(self, prefix:str):
        self.prefix = prefix
        self.frg_delay,self.frg_freq,self.frg_trace = self.load_frog_trace(self.prefix+".A.dat")
        _,_,self.frg_trace_reconstructed = self.load_frog_trace(self.prefix+".Arecon.dat")
        self.time,self.time_intensity,self.time_phase,_,_ = np.loadtxt(self.prefix+".Ek.dat", dtype=float).T
        self.freq,self.freq_intensity,self.freq_phase,_,_ = np.loadtxt(self.prefix+".Ew.dat", dtype=float).T
        self.wavelength,self.wavelength_intensity,self.wavelength_phase,_,_ = np.loadtxt(self.prefix+".Speck.dat", dtype=float).T
        _,_,self.pulse_duration,_ = find_peak_fwhm(self.time,self.time_intensity)
        self.time_FTL,self.time_FTL_intensity = self.get_time_profile_from_frequency(self.freq,self.freq_intensity,0)
        _,_,self.pulse_duration_FTL,_ = find_peak_fwhm(self.time_FTL,self.time_FTL_intensity)
        self._time_reversed = False
        self._base_state = {
            "time": np.asarray(self.time, dtype=float).copy(),
            "time_intensity": np.asarray(self.time_intensity, dtype=float).copy(),
            "time_phase": np.asarray(self.time_phase, dtype=float).copy(),
            "freq": np.asarray(self.freq, dtype=float).copy(),
            "freq_intensity": np.asarray(self.freq_intensity, dtype=float).copy(),
            "freq_phase": np.asarray(self.freq_phase, dtype=float).copy(),
            "wavelength": np.asarray(self.wavelength, dtype=float).copy(),
            "wavelength_intensity": np.asarray(self.wavelength_intensity, dtype=float).copy(),
            "wavelength_phase": np.asarray(self.wavelength_phase, dtype=float).copy(),
            "time_FTL": np.asarray(self.time_FTL, dtype=float).copy(),
            "time_FTL_intensity": np.asarray(self.time_FTL_intensity, dtype=float).copy(),
        }

    def load_frog_trace(self, path:str|PathLike):
        #fread frog data into frog_trace
        """ 
        in matlab, the trace is saved like this
        fprintf(file, ['%d\t%d\n' strf '\t' strf '\n'], sizes, limits);
        fprintf(file, [strf '\n'], lambda);
        fprintf(file, [strf '\n'], delay);
        fprintf(file, [strf '\n'], trace');
        """
        delay,wavelength,freq = [],[],[]
        frog_trace = []
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            #first line: width(N), height(N)
            line1 = lines[0].split()
            if len(line1) < 2:
                raise ValueError(f"Invalid header in {path}: expected at least 2 values, got {len(line1)}")
            width, height = int(line1[0]), int(line1[1])
            
            for line in lines[2:2+height]:
                t = line.split()
                t= [float(t[i]) for i in range(len(t))]
                wavelength.append(t)

            for line in lines[2+height:2+height+width]:
                t = line.split()
                t= [float(t[i]) for i in range(len(t))]
                delay.append(t)

            for line in lines[2+height+width:]:
                t = line.split()
                t= [float(t[i]) for i in range(len(t))]
                frog_trace.append(t)
        freq = speed_light/np.array(wavelength)*1e9/1e15
        frog_trace = np.array(frog_trace)
        return np.array(delay).flatten(), np.array(freq).flatten(), frog_trace.reshape((width,height))

    @staticmethod
    def get_time_profile_from_frequency(freq, intensity, phase):
        # Calculate time profile from frequency profile by inverse Fourier transform.
        freq_arr = np.asarray(freq, dtype=float).ravel()
        intensity_arr = np.asarray(intensity, dtype=float).ravel()
        phase_arr = np.asarray(phase, dtype=float)
        if phase_arr.ndim == 0:
            phase_arr = np.full_like(freq_arr, float(phase_arr), dtype=float)
        else:
            phase_arr = phase_arr.ravel()
        if freq_arr.size != intensity_arr.size or freq_arr.size != phase_arr.size:
            raise ValueError("freq/intensity/phase length mismatch")

        ef = np.sqrt(np.clip(intensity_arr, 0.0, None)) * np.exp(-1.0j * phase_arr)
        time, et = get_Et_from_Ef(freq_arr, ef)
        et = np.asarray(et, dtype=np.complex128).ravel()
        time = np.asarray(time, dtype=float).ravel()
        fft_int = np.abs(et) ** 2
        peak = float(np.max(fft_int)) if fft_int.size else 0.0
        if peak > 1e-15:
            fft_int = fft_int / peak
        return time, fft_int

    def set_time_reversed(self, enabled: bool):
        enabled = bool(enabled)
        if self._time_reversed == enabled:
            return

        b = self._base_state
        if not enabled:
            self.time = b["time"].copy()
            self.time_intensity = b["time_intensity"].copy()
            self.time_phase = b["time_phase"].copy()
            self.freq = b["freq"].copy()
            self.freq_intensity = b["freq_intensity"].copy()
            self.freq_phase = b["freq_phase"].copy()
            self.wavelength = b["wavelength"].copy()
            self.wavelength_intensity = b["wavelength_intensity"].copy()
            self.wavelength_phase = b["wavelength_phase"].copy()
            self.time_FTL = b["time_FTL"].copy()
            self.time_FTL_intensity = b["time_FTL_intensity"].copy()
            self._time_reversed = False
            return

        time = -b["time"]
        idx_t = np.argsort(time)
        self.time = time[idx_t]
        self.time_intensity = b["time_intensity"][idx_t]
        self.time_phase = -b["time_phase"][idx_t]

        self.freq = b["freq"].copy()
        self.freq_intensity = b["freq_intensity"].copy()
        self.freq_phase = -b["freq_phase"].copy()

        self.wavelength = b["wavelength"].copy()
        self.wavelength_intensity = b["wavelength_intensity"].copy()
        self.wavelength_phase = -b["wavelength_phase"].copy()

        time_ftl = -b["time_FTL"]
        idx_ftl = np.argsort(time_ftl)
        self.time_FTL = time_ftl[idx_ftl]
        self.time_FTL_intensity = b["time_FTL_intensity"][idx_ftl]
        self._time_reversed = True

    def get_GDD_TOD(self, freq_min=None, freq_max=None, fit_order=3):
        #calculate GDD and TOD from frequency phase
        #GDD = d2phi/dw2, TOD = d3phi/dw3
        angular_freq = self.freq * 2 * np.pi
        param,_ = fit_peak(angular_freq, self.freq_intensity)
        area,center,sigma = param
        angular_freq0 = center
        _, _, x1, x2 = find_peak_ratio(angular_freq, self.freq_intensity, 0.004)
        if freq_min is None:
            angular_freq_min = x1
        else:
            angular_freq_min = freq_min * 2 * np.pi
        if freq_max is None:
            angular_freq_max = x2
        else:            
            angular_freq_max = freq_max * 2 * np.pi
        idx_min = find_index(angular_freq, angular_freq_min)
        idx_max = find_index(angular_freq, angular_freq_max)
        angular_freq = angular_freq[idx_min:idx_max] - angular_freq0
        phase = self.freq_phase[idx_min:idx_max]
        if len(angular_freq) < 4:
            raise ValueError("Not enough frequency points for phase polynomial fit.")
        fit_order = max(3, min(int(fit_order), len(angular_freq) - 1))
        p = np.polyfit(angular_freq, phase, fit_order)
        phase_fit = np.polyval(p, angular_freq)
        poly = np.poly1d(p)
        GDD = float(np.polyder(poly, 2)(0.0))
        TOD = float(np.polyder(poly, 3)(0.0))
        disp_orders = {n: float(np.polyder(poly, n)(0.0)) for n in range(2, fit_order + 1)}
        return {
            "GDD": GDD,
            "TOD": TOD,
            "disp_orders": disp_orders,
            "phase_fit": phase_fit,
            "angular_freq": angular_freq + angular_freq0,
            "angular_freq0": angular_freq0,
            "phase": phase,
            "fit_order": fit_order,
        }
    
    def dispersion_compensation(self, GDD=0, TOD=0):
        #compensate dispersion by multiplying the phase factor in frequency domain
        angular_freq = self.freq * 2 * np.pi
        angular_freq0 = self.get_GDD_TOD()["angular_freq0"]
        add_phase = GDD / 2 * (angular_freq - angular_freq0) ** 2 + TOD / 6 * (angular_freq - angular_freq0) ** 3
        new_freq_phase = self.freq_phase + add_phase
        new_time, new_time_intensity = self.get_time_profile_from_frequency(self.freq,self.freq_intensity,new_freq_phase)
        return new_time, new_time_intensity
