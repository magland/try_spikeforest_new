import json
import os
import tempfile
import numpy as np
from dataclasses import dataclass
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import kachery_cloud as kcl


# TODO: use dandi api to get the list of files

# for now

@dataclass
class SFRecording:
    study_name: str
    recording_name: str
    nwb_url: str


@dataclass
class SFSorter:
    sorter_name: str


@dataclass
class SFSortingOutput:
    recording: SFRecording
    sorter: SFSorter
    output_object: dict


recordings = [
    # https://neurosift.app/?p=/nwb&url=https://api.dandiarchive.org/api/assets/34a21ebe-c756-4da4-a30b-f1a838a6430b/download/&dandisetId=000618&dandisetVersion=draft
    SFRecording(
        study_name='hybrid-janelia',
        recording_name='hybrid-drift-siprobe-rec-16c-1200s-11',
        nwb_url='https://api.dandiarchive.org/api/assets/34a21ebe-c756-4da4-a30b-f1a838a6430b/download/'
    )
]

sorters = [
    SFSorter(
        sorter_name='kilosort3'
    ),
    SFSorter(
        sorter_name='mountainsort5'
    )
]


def run_all_for_sorter_study(sorter_name, study_name: str):
    sorting_outputs = []
    # find in list
    sorter = next(s for s in sorters if s.sorter_name == sorter_name)
    for recording in recordings:
        if recording.study_name == study_name:
            print(f'Running {sorter_name} for {recording.recording_name}')
            output_object = _run_sorter_on_recording(sorter, recording)
            sorting_outputs.append(SFSortingOutput(
                recording=recording,
                sorter=sorter,
                output_object=output_object
            ))
    a = []
    for output in sorting_outputs:
        a.append({
            'recording_name': output.recording.recording_name,
            'sorter_name': output.sorter.sorter_name,
            'output': output.output_object
        })
    with open(f'output_{sorter_name}_{study_name}.json', 'w') as f:
        json.dump(a, f)


def _run_sorter_on_recording(sorter: SFSorter, recording: SFRecording):
    folder_name = _get_folder_for_sorting_run(sorter, recording)
    output_json_file = f'{folder_name}/output.json'
    if os.path.exists(output_json_file):
        print(f'Output already exists: {output_json_file}')
    else:
        # check if the folder is empty
        if len(os.listdir(folder_name)) > 0:
            raise Exception(f'Folder is not empty: {folder_name}')

        try:
            R = se.read_nwb_recording(recording.nwb_url, stream_mode='remfile')

            ##############################
            # This is needed because the wrong names were used when uploading the spikeforest data to DANDI
            locations = np.array([R.get_property('x'), R.get_property('y')]).T
            R.set_channel_locations(locations)
            ##############################

            R = R.frame_slice(start_frame=0, end_frame=int(R.get_sampling_frequency() * 1))
            S = ss.run_sorter(
                sorter_name=sorter.sorter_name,
                recording=R,
                output_folder=folder_name + '/spikeinterface_output'
            )
            se.NpzSortingExtractor.write_sorting(S, folder_name + '/sorting.npz')
            uri = kcl.store_file(folder_name + '/sorting.npz')
            error_message = None
        except Exception as e:
            print(f'Error running sorter: {e}')
            error_message = str(e)
            uri = None
        with open(output_json_file, 'w') as f:
            json.dump({
                'output_uri': uri,
                'error_message': error_message
            }, f)
    with open(output_json_file, 'r') as f:
        output = json.load(f)
    return output


def _get_folder_for_sorting_run(sorter: SFSorter, recording: SFRecording):
    ret = f'sorting_runs/{recording.study_name}/{recording.recording_name}/{sorter.sorter_name}'
    os.makedirs(ret, exist_ok=True)
    return ret


def test_read_sorting_from_kachery():
    uri = 'sha1://e3b63c6abe3b10d34dee0b166ea20508b50791fb'
    with tempfile.TemporaryDirectory() as tmpdir:
        kcl.load_file(uri, dest=f'{tmpdir}/sorting.npz')
        S = se.NpzSortingExtractor(f'{tmpdir}/sorting.npz')
        print(S)


def collect_ground_truth_for_study(study_name: str):
    output = []
    for recording in recordings:
        if recording.study_name == study_name:
            print(f'Collecting ground truth for {recording.recording_name}')
            S_gt = se.read_nwb_sorting(recording.nwb_url, stream_mode='remfile')
            with tempfile.TemporaryDirectory() as tmpdir:
                se.NpzSortingExtractor.write_sorting(S_gt, f'{tmpdir}/sorting.npz')
                uri = kcl.store_file(f'{tmpdir}/sorting.npz')
                output.append({
                    'recording_name': recording.recording_name,
                    'ground_truth_uri': uri
                })
    with open(f'ground_truth_{study_name}.json', 'w') as f:
        json.dump(output, f)


# TODO: make this a command-line tool to run the various tasks
# each task will produce an output file.
# these output files should get pushed to the repo

if __name__ == '__main__':
    # run_all_for_sorter_study('mountainsort5', 'hybrid-janelia')
    collect_ground_truth_for_study('hybrid-janelia')
    # test_read_sorting_from_kachery()
