"""
Tasks:
- Calculate the irradiance in watts/nm/m^2 for the UpScan and DownScan and compare the results
- Plot the Irradiance as a function of Wavelength around the two emission lines at ~180nm [180 to 183nm]
- Calculate and plot the ratio of the Irradiances at each wavelength for each scan with respect to the reference spectrum
- How do the results compare to the reference spectra?
"""

import datetime

from bokeh.models import Span
from bokeh.plotting import figure, show
import pandas


def apply_scan_regions(plot):
    """
    """

    start_date = datetime.datetime(1980, 1, 6, 0, 0)

    plan_data = pandas.read_csv('data/plans.txt', names=['plan', 'start_time', 'end_time'], header=0)
    downscan, dark, upscan = {}, {}, {}
    for time in ['start_time', 'end_time']:
        downscan[time] = start_date + datetime.timedelta(microseconds=plan_data[time][0])
        dark[time] = start_date + datetime.timedelta(microseconds=plan_data[time][1])
        upscan[time] = start_date + datetime.timedelta(microseconds=plan_data[time][2])

    downscan_start = Span(location=downscan['start_time'], dimension='height', line_color='green', line_dash='dashed', line_width=1)
    downscan_end = Span(location=downscan['end_time'], dimension='height', line_color='green', line_dash='dashed', line_width=1)
    dark_start = Span(location=dark['start_time'], dimension='height', line_color='black', line_dash='dashed', line_width=1)
    dark_end = Span(location=dark['end_time'], dimension='height', line_color='black', line_dash='dashed', line_width=1)
    upscan_start = Span(location=upscan['start_time'], dimension='height', line_color='red', line_dash='dashed', line_width=1)
    upscan_end = Span(location=upscan['end_time'], dimension='height', line_color='red', line_dash='dashed', line_width=1)
    plot.renderers.extend([downscan_start, downscan_end, upscan_start, upscan_end, dark_start, dark_end])

    return plot

def convert_to_datetime(times):
    """
    """

    start_date = datetime.datetime(1980, 1, 6, 0, 0)

    converted_times = []
    for time in times:
        converted_times.append(start_date + datetime.timedelta(microseconds=time))

    return converted_times

if __name__ == '__main__':

    # Make detector temperature plot
    data = pandas.read_csv('data/detectorTemp.txt', names=['time', 'temp'], header=0)
    new_times = convert_to_datetime(data['time'])
    p = figure(title="Detector Temperature", x_axis_label='Time', y_axis_label='Temp (C)', x_axis_type='datetime')
    p.line(new_times, data['temp'], line_width=2)
    p = apply_scan_regions(p)
    show(p)

    # Make Distance and Doppler plot
    data = pandas.read_csv('data/distanceAndDoppler.txt', names=['time', 'distance', 'doppler'], header=0)
    new_times = convert_to_datetime(data['time'])
    p = figure(title="Distance and Doppler", x_axis_label='Time', y_axis_label='Distance', x_axis_type='datetime')
    p.line(new_times, data['distance'], line_width=2)
    p.line(new_times, data['doppler'], line_width=2)
    p = apply_scan_regions(p)
    show(p)

    # Make Grating Position plot
    data = pandas.read_csv('data/instrumentTelemetry.txt', names=['time', 'grating_position', 'counts'], header=0)
    new_times = convert_to_datetime(data['time'])
    p = figure(title="Grating Position", x_axis_label='Time', y_axis_label='Grating Position', x_axis_type='datetime')
    p.line(new_times, data['grating_position'], line_width=2)
    p = apply_scan_regions(p)
    show(p)

    # Make Counts plot
    p = figure(title="Counts", x_axis_label='Time', y_axis_label='Counts', x_axis_type='datetime')
    p.line(new_times, data['counts'], line_width=2)
    p = apply_scan_regions(p)
    show(p)

    # Make Integration Time plot
    data = pandas.read_csv('data/integrationTime.txt', names=['time', 'integration_time'], header=0)
    new_times = convert_to_datetime(data['time'])
    p = figure(title="Integration Time", x_axis_label='Time', y_axis_label='Integration Time (milli-seconds)', x_axis_type='datetime')
    p.line(new_times, data['integration_time'], line_width=2)
    p = apply_scan_regions(p)
    show(p)

    # Make Reference Spectrum plot
    data = pandas.read_csv('data/referenceSpectrum.txt', names=['wavelength', 'irradiance'], header=0)
    p = figure(title="Reference Spectra", x_axis_label='Wavelength (nm)', y_axis_label='Irradiance (watts/m^2/nm)')
    p.line(data['wavelength'], data['irradiance'], line_width=2)
    show(p)

