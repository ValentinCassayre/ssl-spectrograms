import os
from timeit import default_timer as timer
from datetime import datetime, timedelta
from multiprocessing import Process
import numpy as np

from obspy import UTCDateTime
from obspy.clients.fdsn import Client

client = Client("IRIS")


def download_windows_multiprocess(starttime: UTCDateTime, endtime: UTCDateTime, duration=60*60*1, path='data', n_processes=5):
    """
    duration : in seconds
    starttime : UTCDateTime of the beggining of 
    n_processes : number of parallel processes that will be used to download the data 
    according to iris.edu the connections should be limited to 5 (http://ds.iris.edu/ds/nodes/dmc/services/usage/)
    """
    windows = np.arange(starttime, endtime, duration)
    total_windows = len(windows)

    process_start = datetime.now()
    print(process_start.strftime("%d/%m/%Y, %H:%M:%S"))
    print(f'{total_windows=} {n_processes=}')

    window_process_size = total_windows//n_processes
    processes = []
    for k in range(n_processes):

        i1 = k*window_process_size
        if k == n_processes - 1:
            i2 = -1
        else:
            i2 = (k+1)*window_process_size

        process_starttime = windows[i1]
        process_endtime = windows[i2]

        process = Process(target=download_windows, args=(
            process_starttime, process_endtime, duration, path))
        processes.append(process)
        process.start()

        print(f'Creating process {k+1}/{n_processes} : windows {i1} to {i2}')

    for process in processes:
        process.join()

    return


def download_windows(starttime, endtime, duration=60*60*1, path='data'):
    windows = np.arange(starttime, endtime, duration)
    total_windows = len(windows)
    elapseds = []
    failed = 0
    process_start = datetime.now()

    print(f'starttime={starttime.strftime("%d/%m/%Y, %H:%M:%S")} endtime={endtime.strftime("%d/%m/%Y, %H:%M:%S")} process_start={process_start.strftime("%d/%m/%Y, %H:%M:%S")} {total_windows=}')

    for k in range(total_windows):

        start = timer()
        downloaded = download_window(
            windows[k], windows[k] + duration, path=path)
        end = timer()
        elapsed = end - start
        elapseds.append(elapsed)
        process_end_estimate = process_start + \
            timedelta(0, np.average(elapseds) * total_windows)

        if downloaded:
            print(
                f'\nWindow : {windows[k].strftime("%d/%m/%Y, %H:%M:%S")}\nDownloading time : {elapsed}\nEstimated total : {process_end_estimate.strftime("%d/%m/%Y, %H:%M:%S")}')

        else:
            failed += 1

    process_end = datetime.now()
    print(f'starttime={starttime.strftime("%d/%m/%Y, %H:%M:%S")} endtime={endtime.strftime("%d/%m/%Y, %H:%M:%S")} process_end={process_end.strftime("%d/%m/%Y, %H:%M:%S")} {total_windows=} {failed=}')

    return


def download_window(starttime, endtime, skip=True, path='data'):
    filename = f'{starttime.strftime("%Y_%m_%d_%H_%M_%S")}.mseed'
    file = os.path.join(path, filename)

    if skip and os.path.isfile(file):
        return False

    try:
        st = client.get_waveforms(network='G', station='DRV', location='*',
                                  channel='BHZ', starttime=starttime, endtime=endtime)
        st.write(file)
        return True

    except Exception:
        return False


if __name__ == '__main__':
    starttime = UTCDateTime(2000, 1, 1, 0, 0, 0)
    endtime = UTCDateTime(2024, 1, 1, 0, 0, 0)

    # # Single process
    # download_windows(starttime, endtime, duration=60*60*1, path='data')

    # Multiprocessing
    download_windows_multiprocess(
        starttime, endtime, duration=60*60*1, path='data', n_processes=5)
