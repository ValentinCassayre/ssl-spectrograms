import os

import matplotlib

from obspy import UTCDateTime

from download_windows_waveforms import (
    download_inventory,
    download_windows,
    download_windows_multiprocess,
)
from process_spectrogram import (
    read_inventory_xml,
    process_files,
    process_files_multiprocess,
)


if __name__ == "__main__":
    matplotlib.use("Agg")

    starttime = UTCDateTime(2000, 1, 1, 0, 0, 0)
    endtime = UTCDateTime(2024, 1, 1, 0, 0, 0)
    duration = duration = 60 * 60 * 1
    path = "data"
    out_path = "spectrograms"
    inventory_path = "metadata"
    inventory_filename = "inventory.xml"
    n_process = 5

    # Download
    # Inventory
    inventory = download_inventory(inventory_filename=inventory_filename, inventory_path=inventory_path)

    # Single process
    # download_windows(starttime, endtime, duration=duration, path=path)

    # Multiprocessing
    download_windows_multiprocess(starttime, endtime, duration=duration, path=path, n_processes=n_process)

    # Process
    # Inventory
    inventory = read_inventory_xml(inventory_filename=inventory_filename, inventory_path=inventory_path)

    # Files
    files = os.listdir(path)

    # Single process
    # process_files(files, path, inventory=inventory, out_path=out_path)

    # Multiprocessing
    process_files_multiprocess(files, path, inventory=inventory, n_processes=n_process, out_path=out_path)
