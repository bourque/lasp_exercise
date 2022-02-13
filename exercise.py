"""This module contains software for completing the LASP coding
exercise.  The goal of the exercise is to calculate the measured
irradiance from the SOlar Stellar Irradiance Comparison Experiment
(SOLSTICE).

The main tasks of this exercise are to:
    (1) Calculate the irradiance in watts/nm/m^2 for the UpScan and
        DownScan experiments and compare the results
    (2) Plot the irradiance as a function of wavelength around two
        emission lines located at ~180nm
    (3) Calculate and plot the ratio of the irradiances at each
        wavelength for each scan with respect to the reference spectrum

After running this code, a 'plots/' directory will be created and
contain three plots:

    - instrument_data.html : plots the data acquired from the
      instrument
    - irradiance.html : plots the measured irradiance of the UpScan and
      DownScan experiments along with the reference spectrum
    - irradiance_ratios.html : plots the ratio of the UpScan and
      DownScan experiements with respect to the reference spectrum
    - reference_spectrum_fit.html : plots the gaussian fit of the
      Downscan, Upscan, and reference spectrum, which was used to shift
      the wavelength to align with the reference spectrum

Use
---

    To perform these calculations and make corresponding plots:

    From the command line:

        python exercise.py

    Or from within a python interpreter:

        ie = IrradianceExercise()
        ie.get_data()
        ie.make_plots_input_data()
        ie.run_calculations()
        ie.shift_wavelength()
        ie.make_plots_results()

    It is required that a 'data/' directory exists in the working
    directory, and it contains the following data files:

        - detectorTemp.txt : in degrees Celsius. It is roughly sampled
          at 1 second.
        - distanceAndDoppler.txt : These are the corrections used to
          adjust for the changing
          distance and velocity of the spacecraft relative to the sun.
        - instrumentTelemetry.txt : Includes grating position and
          measured detector counts. The detector counts correspond
          to the number of photons detected within the currently set
          integrationTime (in milli-seconds).
        - integrationTime.txt : This is the currently set integration
          time (mlli-seconds) of the instrument. These are sampled
          at a different cadence than the instrumentTelemetry. Assume
          the value is constant until there is a new value.
        - plans.txt : This file includes the experiment names with
          start/end times. You can find the
          time ranges of the plans of interest here.
        - referenceSpectrum.txt : This is a reference spectrum with
          accurate wavelengths. The current irradiance measurements
          will be within 15% of this spectrum.

Dependencies
------------

    - bokeh
    - numpy
    - pandas
    - scipy

    The user may utilize the provided 'requirements.txt' and/or
    'environment.yml' files to create the necessary software
    environment to run the code.  More details about this are provided
    in the README.
"""

from collections import namedtuple
import datetime

from bokeh.layouts import gridplot
from bokeh.models import BoxAnnotation, NumeralTickFormatter
from bokeh.plotting import figure, output_file, save, show
import numpy as np
import pandas
from scipy.optimize import curve_fit


class IrradianceExercise():
    """Main class for completing the exercise"""

    def __init__(self):
        """Initializes a IrradianceExercise class object"""

        # Set some constant values
        self.aperture_Area = .01 / (1E2 * 1E2)  # [m^2]
        self.c = 299792458.0  # [m/s]
        self.corrFactor = 0.0061628
        self.d = 277.77777777777777  # [nm]
        self.detectorTemperatureCorr = 0.0061628
        self.h = 6.62606957E-34  # [J*s]
        self.offset = 239532.38
        self.phiGInRads = 0.08503244115716374  # [rad]
        self.stepSize = 2.4237772022101214E-6  # [rad]

        # A place to store results
        self.results = namedtuple('results', field_names=[])

    def _convert_to_datetime(self, times):
        """Convert a list of times (in microseconds) to a list of
        datetime objects for convenience in plotting.

        Parameters
        ----------
        times : list
            A list of times, e.g. ['9.434207077723681E14', ...]
        """

        converted_times = []
        for time in times:
            converted_times.append(self.data_collection_start + datetime.timedelta(microseconds=time))

        return converted_times

    def _find_closest_measurements(self, data_to_match, data_to_search, data_type):
        """Convenience method to find measurements (e.g. detector
        temperatures, distance corrections) that correspond to the
        given data's time measurements.

        The detector temperature and distance/doppler correction values
        are sampled at a different cadence than other data (e.g. grating
        positions). This method will return a list of measurement
        values that match to the given 'data' as close as possible in
        time.

        Currently, this function only supports searching over detector
        temperature and distance/doppler correction data.

        Parameters
        ----------
        data_to_match : pandas.core.frame.DataFrame object
            The data from which to find the closest values in time
        data_to_search : pandas.core.frame.DataFrame object
            The data to search over (e.g. detector temperatures)
        data_type : str
            The type of data to search over.  Currently only
            'temperature', 'distance_correction', and 'doppler_factor'
            are supported.

        Returns
        -------
        matched_values : np.array object
            The matched temperature values
        """

        matched_values = []
        for index, row in data_to_match.iterrows():
            time_to_match = row['time']
            matched_index = np.argmin(np.abs(data_to_search['time'].values - time_to_match))
            matched_value = data_to_search[data_type].values[matched_index]
            matched_values.append(matched_value)

        return np.array(matched_values)

    def _indicate_experiments(self, plot):
        """Given a bokeh plot, add a shaded region that indicate the
        time span of the downscan, dark, and upscan experiments

        Parameters
        ----------
        plot : bokeh.plotting.figure object
            The bokeh plot to add shaded regions to
        """

        start_times = [self.downscan_start_dt, self.dark_start_dt, self.upscan_start_dt]
        end_times = [self.downscan_end_dt, self.dark_end_dt, self.upscan_end_dt]
        colors = ['green', 'gray', 'red']

        for start, end, color in zip(start_times, end_times, colors):
            plot.add_layout(BoxAnnotation(left=start, right=end, fill_color=color, line_color=color, fill_alpha=0.1))

        return plot

    def calculate_count_rate(self):
        """Calculate the count rate:

        count_rate = counts / integrationTime [counts/s/nm]

        where:
            counts is in [counts/nm]
            integrationTime is in [s]
        """

        # Convert integration times from milliseconds to seconds
        # Also simplify it down to one value since it is constant throughout the experiment
        integration_time_downscan_s = [time * 0.001 for time in self.data.integration_data_downscan['integration_time']][0]
        integration_time_upscan_s = [time * 0.001 for time in self.data.integration_data_upscan['integration_time']][0]

        # Calculate count rates
        count_rates_downscan = self.data.grating_data_downscan['counts'].values / integration_time_downscan_s
        count_rates_upscan = self.data.grating_data_upscan['counts'].values / integration_time_upscan_s

        # Store results for later use in plotting
        self.results.count_rates_downscan = count_rates_downscan
        self.results.count_rates_upscan = count_rates_upscan

    def calculate_count_rate_corr(self):
        """Apply correction to count rate due to changes in temperature:

        count_rate_corr = count_rate * (1.0 + detectorTemperatureCorr * (20.0 - detectorTemp))

        where:
            detectorTemperatureCorr = 0.0061628
        """

        # Find detector temperature measurements closest in time with grading position data
        matched_temperatures_downscan = self._find_closest_measurements(
            self.data.grating_data_downscan, self.data.temperature_data_downscan, 'temperature')
        matched_temperatures_upscan = self._find_closest_measurements(
            self.data.grating_data_upscan, self.data.temperature_data_upscan, 'temperature')

        # Apply temperature correction
        count_rate_corr_downscan = self.results.count_rates_downscan * \
            (1.0 + self.detectorTemperatureCorr * (20.0 - np.array(matched_temperatures_downscan)))
        count_rate_corr_upscan = self.results.count_rates_upscan * \
            (1.0 + self.detectorTemperatureCorr * (20.0 - np.array(matched_temperatures_upscan)))

        # Store results for later use in plotting
        self.results.count_rate_corr_downscan = count_rate_corr_downscan
        self.results.count_rate_corr_upscan = count_rate_corr_upscan

    def calculate_energy_per_photon(self):
        """Calculate the energy per photon:

        energyPerPhotons = h * c / wavelength [J]

        where:
            h = 6.62606957E-34 [J*s]
            c = 299792458.0 [m/s]
            wavelength is in [m]
        """

        # Convert wavelengths from nm to m
        wavelengths_downscan_m = self.results.wavelengths_downscan * 1E-9
        wavelengths_upscan_m = self.results.wavelengths_upscan * 1E-9

        # Calculate energy per photon
        energyPerPhotons_downscan = self.h * self.c / wavelengths_downscan_m
        energyPerPhotons_upscan = self.h * self.c / wavelengths_upscan_m

        # Store results for later use in plotting
        self.results.energyPerPhotons_downscan = energyPerPhotons_downscan
        self.results.energyPerPhotons_upscan = energyPerPhotons_upscan

    def calculate_irradiance(self):
        """Calculate the solar irradiance:

        wattsPerM2_1AU = wattsPerM2 / sunObserverDistanceCorrection

        where:
            wattsPerM2 = photons_per_second_per_m2 * energyPerPhoton
            sunObserverDistanceCorrection is (something)
        """

        # Calculate watts/m^2
        wattsPerM2_downscan = self.results.photons_per_second_per_m2_downscan * self.results.energyPerPhotons_downscan
        wattsPerM2_upscan = self.results.photons_per_second_per_m2_upscan * self.results.energyPerPhotons_upscan

        # Find distance correction measurements closest in time with grating position data
        matched_distances_downscan = self._find_closest_measurements(
            self.data.grating_data_downscan, self.data.distance_data_downscan, 'distance_correction')
        matched_distances_upscan = self._find_closest_measurements(
            self.data.grating_data_upscan, self.data.distance_data_upscan, 'distance_correction')

        # Calculate irradiance
        irradiance_downscan = wattsPerM2_downscan / matched_distances_downscan
        irradiance_upscan = wattsPerM2_upscan / matched_distances_upscan

        # Store results for later use in plotting
        self.results.irradiance_downscan = irradiance_downscan
        self.results.irradiance_upscan = irradiance_upscan

    def calculate_median_dark_count_rate(self):
        """Calculate the median dark count rate, with applying temperature
        correction described in calcualte_count_rate_corr():

        median_dark_count_rate = median(dark_count_rate * ( 1.0 + detectorTemperatureCorr * ( 20.0 - detectorTemp)))

        where:
            dark_count_rate = dark_counts / dark_integrationTime
            detectorTemperatureCorr = 0.0061628
        """

        # Calculate dark count rates
        dark_counts = self.data.grating_data_dark['counts'].values
        dark_integration_times = self.data.integration_data_dark['integration_time'].values
        dark_count_rates = dark_counts / dark_integration_times

        # Find detector temperature measurements closest in time with grating position data
        matched_temperatures_dark = self._find_closest_measurements(
            self.data.grating_data_dark, self.data.temperature_data_dark, 'temperature')

        # Apply temperature correction
        median_dark_count_rate = np.median(dark_count_rates * \
            (1.0 + self.detectorTemperatureCorr * (20.0 - np.array(matched_temperatures_dark))))

        # Store results for later use in plotting
        self.results.median_dark_count_rate = median_dark_count_rate

    def calculate_photons_per_second_per_m2(self):
        """Calculate the number of photons per second per square meter:

        photons_per_second_per_m2 = (count_rate_corr - median_dark_count_rate) / aperture_Area  [photons/sec/m^2/nm]

        where:
            aperture_Area = .01 / (1E2 * 1E2) [m^2]
        """

        # Calculate photons/s/m^2
        aperture_area = .01 / (1E2 * 1E2)
        photons_per_second_per_m2_downscan = (
            self.results.count_rate_corr_downscan - self.results.median_dark_count_rate) / aperture_area
        photons_per_second_per_m2_upscan = (
            self.results.count_rate_corr_upscan - self.results.median_dark_count_rate) / aperture_area

        # Store results for later use in plotting
        self.results.photons_per_second_per_m2_downscan = photons_per_second_per_m2_downscan
        self.results.photons_per_second_per_m2_upscan = photons_per_second_per_m2_upscan

    def calculate_wavelengths(self):
        """Calculate wavelengths (in nm) from the grating positions via
        the grating equation:

        wavelength = 2 * d * sin(ang) * cos(phiGInRads / 2.0) [nm]

        where:
            d = 277.77777777777777 [nm]
            ang = (offset - gratingPosition) * stepSize
            phiGInRads = 0.08503244115716374 [rad]
            offset = 239532.38
            stepSize = 2.4237772022101214E-6 [rad]

        A correction is applied to the wavelength to take the changing
        velocity of the spacecraft into account.
        """

        # Calculate angle of incidence
        angles_downscan = ((self.offset - self.data.grating_data_downscan['grating_position']) * self.stepSize).values
        angles_upscan = ((self.offset - self.data.grating_data_upscan['grating_position']) * self.stepSize).values

        # Calculate wavelength
        wavelengths_downscan = 2 * self.d * np.sin(angles_downscan) * np.cos(self.phiGInRads / 2.0)
        wavelengths_upscan = 2 * self.d * np.sin(angles_upscan) * np.cos(self.phiGInRads / 2.0)

        # Find doppler correction measurements closest in time with wavelength data
        matched_doppler_downscan = self._find_closest_measurements(
            self.data.grating_data_downscan, self.data.distance_data_downscan, 'doppler_factor')
        matched_doppler_upscan = self._find_closest_measurements(
            self.data.grating_data_upscan, self.data.distance_data_upscan, 'doppler_factor')

        # Apply correction for doppler factor
        wavelengths_downscan = wavelengths_downscan / matched_doppler_downscan
        wavelengths_upscan = wavelengths_upscan / matched_doppler_upscan

        # Store results for later use in plotting
        self.results.wavelengths_downscan = wavelengths_downscan
        self.results.wavelengths_upscan = wavelengths_upscan

    def get_data(self):
        """Read in data from input data files and store data within
        class object via a 'data' attribute.  This method also creates
        other attributes (e.g. experiment start/end times, subsets of
        data for specific experiments, etc.) for convenience in
        calculations and/or plotting.
        """

        # Package data into namedtuple object
        self.data = namedtuple('data', field_names=[])
        self.data.temperature_data = pandas.read_csv('data/detectorTemp.txt', names=['time', 'temperature'], header=0)
        self.data.distance_data = pandas.read_csv('data/distanceAndDoppler.txt', names=['time', 'distance_correction', 'doppler_factor'], header=0)
        self.data.grating_data = pandas.read_csv('data/instrumentTelemetry.txt', names=['time', 'grating_position', 'counts'], header=0)
        self.data.integration_data = pandas.read_csv('data/integrationTime.txt', names=['time', 'integration_time'], header=0)
        self.data.reference_spectrum_data = pandas.read_csv('data/referenceSpectrum.txt', names=['wavelength', 'irradiance'], header=0)
        self.data.plan_data = pandas.read_csv('data/plans.txt', names=['plan', 'start_time', 'end_time'], header=0)

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

        # Set some attributes for subsets of data for downscan, dark, and upscan experiments
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

    def make_plots_input_data(self):
        """Create a grid of bokeh plots displaying the data gathered
        from the instrument
        """

        # Make detector temperature plot
        times_datetime = self._convert_to_datetime(self.data.temperature_data['time'])
        temperature_plot = figure(title="Detector Temperature", x_axis_label='Time', y_axis_label='Temp (C)', x_axis_type='datetime')
        temperature_plot.line(times_datetime, self.data.temperature_data['temperature'], line_width=2)
        temperature_plot = self._indicate_experiments(temperature_plot)

        # Make Distance Correction and Doppler Factor plot
        times_datetime = self._convert_to_datetime(self.data.distance_data['time'])
        distance_plot = figure(title="Distance Correction and Doppler Factor", x_axis_label='Time', x_axis_type='datetime')
        distance_plot.line(times_datetime, self.data.distance_data['distance_correction'], line_width=2, line_color='blue', legend_label='Distance Correction')
        distance_plot.line(times_datetime, self.data.distance_data['doppler_factor'], line_width=2, line_color='green', legend_label='Doppler Factor')
        distance_plot.legend.location = 'right'
        distance_plot = self._indicate_experiments(distance_plot)

        # Make Grating Position plot
        times_datetime = self._convert_to_datetime(self.data.grating_data['time'])
        grating_position_plot = figure(title="Grating Position", x_axis_label='Time', x_axis_type='datetime')
        grating_position_plot.line(times_datetime, self.data.grating_data['grating_position'], line_width=2)
        grating_position_plot = self._indicate_experiments(grating_position_plot)
        grating_position_plot.yaxis[0].formatter = NumeralTickFormatter(format="0")

        # Make Counts plot
        counts_plot = figure(title="Counts", x_axis_label='Time', x_axis_type='datetime')
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
        reference_plot.yaxis[0].formatter = NumeralTickFormatter(format="0.000")

        # Arrange plots in a grid
        grid = gridplot([[temperature_plot, distance_plot, grating_position_plot],
                         [counts_plot, integration_time_plot, reference_plot]], width=500, height=500)

        # Save plot
        filename = 'plots/instrument_data.html'
        output_file(filename=filename)
        save(grid)
        print(f'\tPlot saved to {filename}')

    def make_plots_results(self):
        """Create bokeh plots showing results of the exercise"""

        # Plot irradiance near ~180nm emission lines
        p = figure(title="Irradiance at 1AU", x_axis_label='Wavelength (nm)', y_axis_label='Irradiance (watts/m^2/nm)', x_range=(180, 183))
        p.line(self.results.wavelengths_downscan, self.results.irradiance_downscan, line_width=2, color='green', legend_label='Downscan')
        p.line(self.results.wavelengths_upscan, self.results.irradiance_upscan, line_width=2, color='red', legend_label='Upscan')
        p.line(self.data.reference_spectrum_data['wavelength'], self.data.reference_spectrum_data['irradiance'], line_width=2, line_color='black', legend_label='Reference')
        filename = 'plots/irradiance.html'
        output_file(filename=filename)
        save(p)
        print(f'\tPlot saved to {filename}')

        # Find the reference spectrum values that match downscan/upscan wavelengths the closest
        reference_values_downscan, reference_values_upscan = [], []
        for index, wavelength_to_match in enumerate(self.results.wavelengths_downscan):
            matched_index = np.argmin(np.abs(self.data.reference_spectrum_data['wavelength'].values - wavelength_to_match))
            matched_value = self.data.reference_spectrum_data['irradiance'].values[matched_index]
            reference_values_downscan.append(matched_value)
        for index, wavelength_to_match in enumerate(self.results.wavelengths_upscan):
            matched_index = np.argmin(np.abs(self.data.reference_spectrum_data['wavelength'].values - wavelength_to_match))
            matched_value = self.data.reference_spectrum_data['irradiance'].values[matched_index]
            reference_values_upscan.append(matched_value)

        # Plot irradiance ratio w.r.t. reference
        downscan_ratio = self.results.irradiance_downscan / reference_values_downscan
        upscan_ratio = self.results.irradiance_upscan / reference_values_upscan
        p = figure(title="Irradiance Ratio w.r.t. Reference Spectra", x_axis_label='Wavelength (nm)', y_axis_label='Irradiance (watts/m^2/nm)', x_range=(180, 183))
        p.line(self.results.wavelengths_downscan, downscan_ratio, line_width=2, color='green', legend_label='Downscan')
        p.line(self.results.wavelengths_upscan, upscan_ratio, line_width=2, color='red', legend_label='Upscan')
        filename = 'plots/irradiance_ratios.html'
        output_file(filename=filename)
        save(p)
        print(f'\tPlot saved to {filename}')

    def run_calculations(self):
        """Main method for running calculations necessary for
        determining irradiance
        """

        self.calculate_wavelengths()
        self.calculate_energy_per_photon()
        self.calculate_count_rate()
        self.calculate_count_rate_corr()
        self.calculate_median_dark_count_rate()
        self.calculate_photons_per_second_per_m2()
        self.calculate_irradiance()

    def shift_wavelengths(self):
        """Perform a fit and an interpolation of the reference spectrum
        and shift the measured wavelengths to match the spectrum
        """

        # Define gaussian function used for fitting
        def gauss(x, a, x0, sigma):
            return a*np.exp(-(x-x0)**2/(2*sigma**2))

        # Use subsets of data around the ~181.7 nm emission line
        ref_subset = self.data.reference_spectrum_data[self.data.reference_spectrum_data['wavelength'].between(181.62, 181.78)]
        ref_subset_wavelength, ref_subset_irradiance = ref_subset['wavelength'].values, ref_subset['irradiance'].values
        downscan_subset_wavelength = self.results.wavelengths_downscan[np.where((self.results.wavelengths_downscan >= 181.63) & (self.results.wavelengths_downscan <= 181.80))]
        downscan_subset_irradiance = self.results.irradiance_downscan[np.where((self.results.wavelengths_downscan >= 181.63) & (self.results.wavelengths_downscan <= 181.80))]
        upscan_subset_wavelength = self.results.wavelengths_upscan[np.where((self.results.wavelengths_upscan >= 181.64) & (self.results.wavelengths_upscan <= 181.82))]
        upscan_subset_irradiance = self.results.irradiance_upscan[np.where((self.results.wavelengths_upscan >= 181.64) & (self.results.wavelengths_upscan <= 181.82))]

        # Use the max values as a proxy for initial guesses
        guess_ref = ref_subset_wavelength[np.argmax(ref_subset_irradiance)]
        guess_downscan = downscan_subset_wavelength[np.argmax(downscan_subset_irradiance)]
        guess_upscan = upscan_subset_wavelength[np.argmax(upscan_subset_irradiance)]

        # Fit a guassian to the emission line
        params_ref, _ = curve_fit(gauss, ref_subset_wavelength, ref_subset_irradiance, p0=[0.1, guess_ref, 0.1])
        fit_ref = gauss(ref_subset_wavelength, params_ref[0], params_ref[1], params_ref[2])
        params_downscan, _ = curve_fit(gauss, downscan_subset_wavelength, downscan_subset_irradiance, p0=[0.1, guess_downscan, 0.1])
        fit_downscan = gauss(downscan_subset_wavelength, params_downscan[0], params_downscan[1], params_downscan[2])
        params_upscan, _ = curve_fit(gauss, upscan_subset_wavelength, upscan_subset_irradiance, p0=[0.1, guess_upscan, 0.1])
        fit_upscan = gauss(upscan_subset_wavelength, params_upscan[0], params_upscan[1], params_upscan[2])

        # Find the wavelength at the peak of the emission line
        wavelength_at_peak_reference = ref_subset_wavelength[np.argmax(fit_ref)]
        wavelength_at_peak_downscan = downscan_subset_wavelength[np.argmax(fit_downscan)]
        wavelength_at_peak_upscan = upscan_subset_wavelength[np.argmax(fit_upscan)]

        # Find the differences between peaks
        downscan_shift = wavelength_at_peak_downscan - wavelength_at_peak_reference
        upscan_shift = wavelength_at_peak_upscan - wavelength_at_peak_reference

        # Plot the fits for visual inspection
        p = figure(title="Irradiance at 1AU", x_axis_label='Wavelength (nm)', y_axis_label='Irradiance (watts/m^2/nm)', x_range=(181.5, 181.9))
        p.line(self.data.reference_spectrum_data['wavelength'], self.data.reference_spectrum_data['irradiance'], line_width=2, line_color='black', legend_label='Reference')
        p.line(ref_subset_wavelength, fit_ref, line_width=1, color='black', line_dash='dashed', legend_label='Reference fit')
        p.line(self.results.wavelengths_downscan, self.results.irradiance_downscan, line_width=2, color='green', legend_label='Downscan')
        p.line(downscan_subset_wavelength, fit_downscan, line_width=1, color='green', line_dash='dashed', legend_label='Downscan fit')
        p.line(self.results.wavelengths_upscan, self.results.irradiance_upscan, line_width=2, color='red', legend_label='Upscan')
        p.line(upscan_subset_wavelength, fit_upscan, line_width=1, color='red', line_dash='dashed', legend_label='Upscan fit')
        filename = 'plots/reference_spectrum_fit.html'
        output_file(filename=filename)
        save(p)
        print(f'\tPlot saved to {filename}')

        # Apply wavelength shift
        self.results.wavelengths_downscan -= downscan_shift
        self.results.wavelengths_upscan -= upscan_shift


if __name__ == '__main__':

    ie = IrradianceExercise()
    ie.get_data()
    ie.make_plots_input_data()
    ie.run_calculations()
    ie.shift_wavelengths()
    ie.make_plots_results()
