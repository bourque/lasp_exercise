"""
Tasks:
- Calculate the irradiance in watts/nm/m^2 for the UpScan and DownScan and compare the results
- Plot the Irradiance as a function of Wavelength around the two emission lines at ~180nm [180 to 183nm]
- Calculate and plot the ratio of the Irradiances at each wavelength for each scan with respect to the reference spectrum
- How do the results compare to the reference spectra?

Equtions for reference:
    count_rate = counts / integrationTime # [counts / sec / nm]
    detectorTemperatureCorr = 0.0061628
    count_rate_corr = count_rate * (1.0 + detectorTemperatureCorr * (20.0 - detectorTemp))
    dark_count_rate = dark_counts / dark_integrationTime
    median_dark_count_rate = median(dark_count_rate * ( 1.0 + detectorTemperatureCorr * ( 20.0 - detectorTemp)))
    aperture_Area = .01 / (1E2 * 1E2) # [m^2]
    photonsPerSecondPerM2 = (count_rate_corr - median_dark_count_rate) / aperture_Area

    corrected_rate = count_rate * (1 + corrFactor * (referenceTemp - detectorTemp))
    corrFactor = 0.0061628

    offset = 239532.38
    stepSize = 2.4237772022101214E-6 # [rad]
    d = 277.77777777777777 # [nm]
    phiGInRads = 0.08503244115716374 # [rad]
    gratingPosition = 'Get from data'
    ang1 = (offset - gratingPosition) * stepSize
    wavelength = 2 * d * sin(ang1) * cos(phiGInRads / 2.0) # [nm]

    h = 6.62606957E-34  # [J*s]
    c = 299792458.0  # [m/s]
    wavelengthInMeters = 'Get from equation above after converting'
    energyPerPhoton = h * c / wavelengthInMeters

    wattsPerM2 = photonsPerSecondPerM2 * energyPerPhoton
    irradiance = wattsPerM2_1AU = wattsPerM2 / sunObserverDistanceCorrection

To do:
    - Make plots for results
    - Try to make code more DRY
    - Fix/implement remaining corrections
    - Write unit tests
    - Add docstrings
    - Implement CI
    - PEP8 changes
    - Finalize conda env and/or requirements file
    - Document possible improvements
    - Write README and/or report
    - Set up PR
"""

from collections import namedtuple
import datetime
import math

from bokeh.layouts import gridplot
from bokeh.models import BoxAnnotation, Span
from bokeh.plotting import figure, output_file, show
import numpy as np
import pandas

class IrradianceExercise():
    """
    """

    def __init__(self):
        """
        """

        # Set some constant values
        self.aperture_Area = .01 / (1E2 * 1E2) # [m^2]
        self.c = 299792458.0  # [m/s]
        self.corrFactor = 0.0061628
        self.d = 277.77777777777777 # [nm]
        self.detectorTemperatureCorr = 0.0061628
        self.h = 6.62606957E-34  # [J*s]
        self.offset = 239532.38
        self.phiGInRads = 0.08503244115716374 # [rad]
        self.stepSize = 2.4237772022101214E-6 # [rad]

        # Read in data
        self._get_data()

        # Set some other attributes for convenience
        self.data_collection_start = datetime.datetime(1980, 1, 6, 0, 0)
        self.downscan_start = self.data.plan_data['start_time'][0]
        self.downscan_end = self.data.plan_data['end_time'][0]
        self.dark_start = self.data.plan_data['start_time'][1]
        self.dark_end = self.data.plan_data['end_time'][1]
        self.upscan_start = self.data.plan_data['start_time'][2]
        self.upscan_end = self.data.plan_data['end_time'][2]
        self.downscan_start_dt = self.data_collection_start + datetime.timedelta(microseconds=self.downscan_start)
        self.downscan_end_dt = self.data_collection_start + datetime.timedelta(microseconds=self.downscan_end)
        self.dark_start_dt = self.data_collection_start + datetime.timedelta(microseconds=self.dark_start)
        self.dark_end_dt = self.data_collection_start + datetime.timedelta(microseconds=self.dark_end)
        self.upscan_start_dt = self.data_collection_start + datetime.timedelta(microseconds=self.upscan_start)
        self.upscan_end_dt = self.data_collection_start + datetime.timedelta(microseconds=self.upscan_end)

        # Make useful subsets of data based on experiement
        self._make_data_subsets()

        # A place to store results
        self.results = namedtuple('results', field_names=[])


    def _convert_to_datetime(self, times):
        """
        """

        converted_times = []
        for time in times:
            converted_times.append(self.data_collection_start + datetime.timedelta(microseconds=time))

        return converted_times

    def _get_data(self):
        """
        """

        # Package data into namedtuple object
        self.data = namedtuple('data', field_names=[])
        self.data.temperature_data = pandas.read_csv('data/detectorTemp.txt', names=['time', 'temperature'], header=0)
        self.data.distance_data = pandas.read_csv('data/distanceAndDoppler.txt', names=['time', 'distance_correction', 'doppler_factor'], header=0)
        self.data.grating_data = pandas.read_csv('data/instrumentTelemetry.txt', names=['time', 'grating_position', 'counts'], header=0)
        self.data.integration_data = pandas.read_csv('data/integrationTime.txt', names=['time', 'integration_time'], header=0)
        self.data.reference_spectrum_data = pandas.read_csv('data/referenceSpectrum.txt', names=['wavelength', 'irradiance'], header=0)
        self.data.plan_data = pandas.read_csv('data/plans.txt', names=['plan', 'start_time', 'end_time'], header=0)

    def _make_data_subsets(self):
        """
        """

        self.data.temperature_data_downscan = self.data.temperature_data[self.data.temperature_data['time'].between(self.downscan_start, self.downscan_end)]
        self.data.temperature_data_dark = self.data.temperature_data[self.data.temperature_data['time'].between(self.dark_start, self.dark_end)]
        self.data.temperature_data_upscan = self.data.temperature_data[self.data.temperature_data['time'].between(self.upscan_start, self.upscan_end)]

        self.data.distance_data_downscan = self.data.distance_data[self.data.distance_data['time'].between(self.downscan_start, self.downscan_end)]
        self.data.distance_data_dark = self.data.distance_data[self.data.distance_data['time'].between(self.dark_start, self.dark_end)]
        self.data.distance_data_upscan = self.data.distance_data[self.data.distance_data['time'].between(self.upscan_start, self.upscan_end)]

        self.data.grating_data_downscan = self.data.grating_data[self.data.grating_data['time'].between(self.downscan_start, self.downscan_end)]
        self.data.grating_data_dark = self.data.grating_data[self.data.grating_data['time'].between(self.dark_start, self.dark_end)]
        self.data.grating_data_upscan = self.data.grating_data[self.data.grating_data['time'].between(self.upscan_start, self.upscan_end)]

        self.data.integration_data_downscan = self.data.integration_data[self.data.integration_data['time'].between(self.downscan_start, self.downscan_end)]
        self.data.integration_data_dark = self.data.integration_data[self.data.integration_data['time'].between(self.dark_start, self.dark_end)]
        self.data.integration_data_upscan = self.data.integration_data[self.data.integration_data['time'].between(self.upscan_start, self.upscan_end)]

    def _indicate_experiments(self, plot):
        """
        """

        # Mark experiment times with shaded regions
        plot.add_layout(BoxAnnotation(left=self.downscan_start_dt, right=self.downscan_end_dt, fill_alpha=0.1, fill_color='green', line_color='green'))
        plot.add_layout(BoxAnnotation(left=self.dark_start_dt, right=self.dark_end_dt, fill_alpha=0.1, fill_color='gray', line_color='black'))
        plot.add_layout(BoxAnnotation(left=self.upscan_start_dt, right=self.upscan_end_dt, fill_alpha=0.1, fill_color='red', line_color='red'))

        return plot

    def calculate_wavelengths(self):
        """
        """

        angs_downscan = ((self.offset - self.data.grating_data_downscan['grating_position']) * self.stepSize).values
        angs_upscan = ((self.offset - self.data.grating_data_upscan['grating_position']) * self.stepSize).values

        wavelengths_downscan = 2 * self.d * np.sin(angs_downscan) * np.cos(self.phiGInRads / 2.0)
        wavelengths_upscan = 2 * self.d * np.sin(angs_upscan) * np.cos(self.phiGInRads / 2.0)

        # Store results for later use in plotting
        self.results.wavelengths_downscan = wavelengths_downscan # nm
        self.results.wavelengths_upscan = wavelengths_upscan # nm

    def calculate_energyPerPhoton(self):
        """
        """

        # Convert wavelengths from nm to m
        wavelengths_downscan_m = self.results.wavelengths_downscan * 1E-9
        wavelengths_upscan_m = self.results.wavelengths_upscan * 1E-9

        # Calculate energy per photon
        energyPerPhotons_downscan = self.h * self.c / wavelengths_downscan_m
        energyPerPhotons_upscan = self.h * self.c / wavelengths_upscan_m

        # Store results for later use in plotting
        self.results.energyPerPhotons_downscan = energyPerPhotons_downscan # J
        self.results.energyPerPhotons_upscan = energyPerPhotons_upscan # J

    def calculate_count_rate(self):
        """
        """

        counts_downscan = self.data.grating_data_downscan['counts'].values
        counts_upscan = self.data.grating_data_upscan['counts'].values
        integration_times_downscan = self.data.integration_data_downscan['integration_time']
        integration_times_upscan = self.data.integration_data_upscan['integration_time']

        # Convert integration times from milliseconds to seconds
        # Also simplyfy it down to one value since it is constant throughout the experiement
        integration_time_downscan = [time * 0.001 for time in integration_times_downscan][0]
        integration_time_upscan = [time * 0.001 for time in integration_times_upscan][0]

        count_rates_downscan = counts_downscan / integration_time_downscan
        count_rates_upscan = counts_upscan / integration_time_upscan

        # Store results for later use in plotting
        self.results.count_rates_downscan = count_rates_downscan # counts/sec/nm
        self.results.count_rates_upscan = count_rates_upscan  # counts/sec/nm

    def calculate_count_rate_corr(self):
        """
        """

        temperatures_downscan = self.data.temperature_data_downscan['temperature']
        temperatures_upscan = self.data.temperature_data_upscan['temperature']

        # # For each count value, find corresponding temperature value
        # chosen_temperatures_downscan, chosen_temperatures_upscan = [], []
        # grating_data_times_datetime = self._convert_to_datetime(self.data.grating_data['time'])
        # temperature_data_times_datetime = self._convert_to_datetime(self.data.temperature_data['time'])
        # temperature_data_times_datetime = [item.replace(microsecond=0) for item in temperature_data_times_datetime]
        # for time, count in zip(self.data.grating_data_downscan['time'], self.data.grating_data_downscan['counts'].values):
        #     time_datetime = self.data_collection_start + datetime.timedelta(microseconds=time)
        #     time_datetime = time_datetime.replace(microsecond=0)
        #     index = [i for i, time in enumerate(temperature_data_times_datetime) if time == time_datetime]
        #     print(temperatures_downscan[index])
        # temp workaround
        temperatures_downscan = temperatures_downscan[0:2528]
        temperatures_upscan = temperatures_upscan[0:2517]

        count_rate_corr_downscan = self.results.count_rates_downscan * (1.0 + self.detectorTemperatureCorr * (20.0 - temperatures_downscan))
        count_rate_corr_upscan = self.results.count_rates_upscan * (1.0 + self.detectorTemperatureCorr * (20.0 - temperatures_upscan))

        # Store results for later use in plotting
        self.results.count_rate_corr_downscan = count_rate_corr_downscan  # counts/sec/nm
        self.results.count_rate_corr_upscan = count_rate_corr_upscan  # counts/sec/nm

    def calculate_median_dark_count_rate(self):
        """
        """

        dark_counts = self.data.grating_data_dark['counts'].values
        dark_integration_times = self.data.integration_data_dark['integration_time'].values
        dark_temperatures = self.data.temperature_data_dark['temperature'].values

        dark_count_rates = dark_counts / dark_integration_times

        # temp fix to get around indices problems
        dark_count_rates = dark_count_rates[0:-1]
        median_dark_count_rate = np.median(dark_count_rates * (1.0 + self.detectorTemperatureCorr * (20.0 - dark_temperatures)))
        #median_dark_count_rates = [np.median(dark_count_rate * (1.0 + self.detectorTemperatureCorr * (20.0 - temp))) for dark_count_rate, temp in zip(dark_count_rates, dark_temperatures)]

        # Store results for later use in plotting
        self.results.median_dark_count_rate = median_dark_count_rate  # counts/sec/nm

    def calculate_photonsPerSecondPerM2(self):
        """
        """

        aperture_area = .01 / (1E2 * 1E2) # [m^2]
        photonsPerSecondPerM2_downscan = (self.results.count_rate_corr_downscan - self.results.median_dark_count_rate) / aperture_area
        photonsPerSecondPerM2_upscan = (self.results.count_rate_corr_upscan - self.results.median_dark_count_rate) / aperture_area

        # units issue?
        # counts/sec/m / m^2

        # Store results for later use in plotting
        self.results.photonsPerSecondPerM2_downscan = photonsPerSecondPerM2_downscan  # counts/sec/m^2
        self.results.photonsPerSecondPerM2_upscan = photonsPerSecondPerM2_upscan  # counts/sec/m^2

    def calculate_irradiance(self):
        """
        """

        wattsPerM2_downscan = self.results.photonsPerSecondPerM2_downscan * self.results.energyPerPhotons_downscan
        wattsPerM2_upscan = self.results.photonsPerSecondPerM2_upscan * self.results.energyPerPhotons_upscan

        # Interpolate distance correction
        #irradiance_downscan = wattsPerM2_downscan / self.data.distance_data_downscan['distance_correction'].values
        #irradiance_upscan = wattsPerM2_upscan / self.data.distance_data_upscan['distance_correction'].values

        # Make plots
        p = figure(title="WattsPerM2", x_axis_label='Wavelength', y_axis_label='WattsPerM2')
        p.line(self.results.wavelengths_downscan, wattsPerM2_downscan, line_width=2, color='green')
        p.line(self.results.wavelengths_upscan, wattsPerM2_upscan, line_width=2, color='red')
        p.line(self.data.reference_spectrum_data['wavelength'], self.data.reference_spectrum_data['irradiance'], line_width=2, line_color='purple')
        show(p)

    def make_plots_raw_data(self):
        """
        """

        # Make detector temperature plot
        times_datetime = self._convert_to_datetime(self.data.temperature_data['time'])
        temperature_plot = figure(title="Detector Temperature", x_axis_label='Time', y_axis_label='Temp (C)', x_axis_type='datetime')
        temperature_plot.line(times_datetime, self.data.temperature_data['temperature'], line_width=2)
        temperature_plot = self._indicate_experiments(temperature_plot)

        # Make Distance and Doppler plot
        times_datetime = self._convert_to_datetime(self.data.distance_data['time'])
        distance_plot = figure(title="Distance Correction and Doppler Factor", x_axis_label='Time', y_axis_label='Distance', x_axis_type='datetime')
        distance_plot.line(times_datetime, self.data.distance_data['distance_correction'], line_width=2)
        distance_plot.line(times_datetime, self.data.distance_data['doppler_factor'], line_width=2)
        distance_plot = self._indicate_experiments(distance_plot)

        # Make Grating Position plot
        times_datetime = self._convert_to_datetime(self.data.grating_data['time'])
        grating_position_plot = figure(title="Grating Position", x_axis_label='Time', y_axis_label='Grating Position', x_axis_type='datetime')
        grating_position_plot.line(times_datetime, self.data.grating_data['grating_position'], line_width=2)
        grating_position_plot = self._indicate_experiments(grating_position_plot)

        # Make Counts plot
        counts_plot = figure(title="Counts", x_axis_label='Time', y_axis_label='Counts', x_axis_type='datetime')
        counts_plot.line(times_datetime, self.data.grating_data['counts'], line_width=2)
        counts_plot = self._indicate_experiments(counts_plot)

        # Make Integration Time plot
        times_datetime = self._convert_to_datetime(self.data.integration_data['time'])
        integration_time_plot = figure(title="Integration Time", x_axis_label='Time', y_axis_label='Integration Time (milli-seconds)', x_axis_type='datetime')
        integration_time_plot.line(times_datetime, self.data.integration_data['integration_time'], line_width=2)
        integration_time_plot = self._indicate_experiments(integration_time_plot)

        # Make Reference Spectrum plot
        reference_plot = figure(title="Reference Spectra", x_axis_label='Wavelength (nm)', y_axis_label='Irradiance (watts/m^2/nm)')
        reference_plot.line(self.data.reference_spectrum_data['wavelength'], self.data.reference_spectrum_data['irradiance'], line_width=2)

        grid = gridplot([[temperature_plot, distance_plot, grating_position_plot],
                         [counts_plot, integration_time_plot, reference_plot]], width=500, height=500)
        show(grid)

    def make_plots_results(self):
        """
        """

        pass

if __name__ == '__main__':

    irradiance_exercise = IrradianceExercise()
    irradiance_exercise.make_plots_raw_data()
    irradiance_exercise.calculate_wavelengths()
    irradiance_exercise.calculate_energyPerPhoton()
    irradiance_exercise.calculate_count_rate()
    irradiance_exercise.calculate_count_rate_corr()
    irradiance_exercise.calculate_median_dark_count_rate()
    irradiance_exercise.calculate_photonsPerSecondPerM2()
    irradiance_exercise.calculate_irradiance()
    irradiance_exercise.make_plots_results()
