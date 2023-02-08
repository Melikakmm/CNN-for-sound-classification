
import pydot
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ctypes
import shutil
import torch
from torchaudio.transforms import PitchShift, TimeStretch

import os.path
import ast


# Number of samples per 30s audio clip.
# TODO: fix dataset to be constant.
NB_AUDIO_SAMPLES = 1321967
SAMPLING_RATE = 44100


class FreeMusicArchive:

    BASE_URL = 'https://freemusicarchive.org/api/get/'

    def __init__(self, api_key):
        self.api_key = api_key

    def get_recent_tracks(self):
        URL = 'https://freemusicarchive.org/recent.json'
        r = requests.get(URL)
        r.raise_for_status()
        tracks = []
        artists = []
        date_created = []
        for track in r.json()['aTracks']:
            tracks.append(track['track_id'])
            artists.append(track['artist_name'])
            date_created.append(track['track_date_created'])
        return tracks, artists, date_created

    def _get_data(self, dataset, fma_id, fields=None):
        url = self.BASE_URL + dataset + 's.json?'
        url += dataset + '_id=' + str(fma_id) + '&api_key=' + self.api_key
        # print(url)
        r = requests.get(url)
        r.raise_for_status()
        if r.json()['errors']:
            raise Exception(r.json()['errors'])
        data = r.json()['dataset'][0]
        r_id = data[dataset + '_id']
        if r_id != str(fma_id):
            raise Exception('The received id {} does not correspond to'
                            'the requested one {}'.format(r_id, fma_id))
        if fields is None:
            return data
        if type(fields) is list:
            ret = {}
            for field in fields:
                ret[field] = data[field]
            return ret
        else:
            return data[fields]

    def get_track(self, track_id, fields=None):
        return self._get_data('track', track_id, fields)

    def get_album(self, album_id, fields=None):
        return self._get_data('album', album_id, fields)

    def get_artist(self, artist_id, fields=None):
        return self._get_data('artist', artist_id, fields)

    def get_all(self, dataset, id_range):
        index = dataset + '_id'

        id_ = 2 if dataset == 'track' else 1
        row = self._get_data(dataset, id_)
        df = pd.DataFrame(columns=row.keys())
        df.set_index(index, inplace=True)

        not_found_ids = []

        for id_ in id_range:
            try:
                row = self._get_data(dataset, id_)
            except:
                not_found_ids.append(id_)
                continue
            row.pop(index)
            df = df.append(pd.Series(row, name=id_))

        return df, not_found_ids

    def download_track(self, track_file, path):
        url = 'https://files.freemusicarchive.org/' + track_file
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

    def get_track_genres(self, track_id):
        genres = self.get_track(track_id, 'track_genres')
        genre_ids = []
        genre_titles = []
        for genre in genres:
            genre_ids.append(genre['genre_id'])
            genre_titles.append(genre['genre_title'])
        return genre_ids, genre_titles

    def get_all_genres(self):
        df = pd.DataFrame(columns=['genre_parent_id', 'genre_title',
                                   'genre_handle', 'genre_color'])
        df.index.rename('genre_id', inplace=True)

        page = 1
        while True:
            url = self.BASE_URL + 'genres.json?limit=50'
            url += '&page={}&api_key={}'.format(page, self.api_key)
            r = requests.get(url)
            for genre in r.json()['dataset']:
                genre_id = int(genre.pop(df.index.name))
                df.loc[genre_id] = genre
            assert (r.json()['page'] == str(page))
            page += 1
            if page > r.json()['total_pages']:
                break

        return df


class Genres:

    def __init__(self, genres_df):
        self.df = genres_df

    def create_tree(self, roots, depth=None):

        if type(roots) is not list:
            roots = [roots]
        graph = pydot.Dot(graph_type='digraph', strict=True)

        def create_node(genre_id):
            title = self.df.at[genre_id, 'title']
            ntracks = self.df.at[genre_id, '#tracks']
            # name = self.df.at[genre_id, 'title'] + '\n' + str(genre_id)
            name = '"{}\n{} / {}"'.format(title, genre_id, ntracks)
            return pydot.Node(name)

        def create_tree(root_id, node_p, depth):
            if depth == 0:
                return
            children = self.df[self.df['parent'] == root_id]
            for child in children.iterrows():
                genre_id = child[0]
                node_c = create_node(genre_id)
                graph.add_edge(pydot.Edge(node_p, node_c))
                create_tree(genre_id, node_c,
                            depth-1 if depth is not None else None)

        for root in roots:
            node_p = create_node(root)
            graph.add_node(node_p)
            create_tree(root, node_p, depth)

        return graph

    def find_roots(self):
        roots = []
        for gid, row in self.df.iterrows():
            parent = row['parent']
            title = row['title']
            if parent == 0:
                roots.append(gid)
            elif parent not in self.df.index:
                msg = '{} ({}) has parent {} which is missing'.format(
                        gid, title, parent)
                raise RuntimeError(msg)
        return roots


def load(filepath):

    filename = os.path.basename(filepath)

    if 'features' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'echonest' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'genres' in filename:
        return pd.read_csv(filepath, index_col=0)

    if 'tracks' in filename:
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

        COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                   ('track', 'genres'), ('track', 'genres_all')]
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)

        COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                   ('album', 'date_created'), ('album', 'date_released'),
                   ('artist', 'date_created'), ('artist', 'active_year_begin'),
                   ('artist', 'active_year_end')]
        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])

        SUBSETS = ('small', 'medium', 'large')
        try:
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                    'category', categories=SUBSETS, ordered=True)
        except (ValueError, TypeError):
            # the categories and ordered arguments were removed in pandas 0.25
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                     pd.CategoricalDtype(categories=SUBSETS, ordered=True))

        COLUMNS = [('track', 'genre_top'), ('track', 'license'),
                   ('album', 'type'), ('album', 'information'),
                   ('artist', 'bio')]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype('category')

        return tracks


def get_audio_path(audio_dir, track_id):
    """
    Return the path to the mp3 given the directory where the audio is stored
    and the track ID.

    Examples
    --------
    >>> import utils
    >>> AUDIO_DIR = os.environ.get('AUDIO_DIR')
    >>> utils.get_audio_path(AUDIO_DIR, 2)
    '../data/fma_small/000/000002.mp3'

    """
    tid_str = '{:06d}'.format(track_id)
    return os.path.join(audio_dir, tid_str[:3], tid_str + '.mp3')


def mono_to_stereo(waveform):
    # reshape the waveform from a 1D tensor to a 2D tensor with 2 columns
    waveform = waveform.view(-1, 1)
    # repeat the waveform along the columns to create a stereo signal
    waveform = waveform.repeat(1, 2)
    return waveform.T


def stereo_to_mono(waveform):
    if waveform.dim() == 2:
        return torch.mean(waveform, dim=0)
    elif waveform.dim() == 1:
        return waveform
    else:
        raise ValueError("Input must be 1D or 2D tensor")
        
        

def frequency_mask(spectrogram, F=15, num_masks=1, replace_with_zero=True):
    """Applies frequency mask on the spectrogram
    Args:
        spectrogram: numpy array, input spectrogram
        F: int, maximum size of each mask
        num_masks: int, number of masks to apply
        replace_with_zero: bool, replace masked values with zeros or not
    Returns:
        masked_spectrogram: numpy array, spectrogram with frequency masks applied
    """
    masked_spectrogram = spectrogram.clone()
    num_mel_channels = spectrogram.shape[0]
    for i in range(num_masks):
        f = np.random.randint(0, num_mel_channels - F)
        f0 = f
        f1 = f + F
        if replace_with_zero:
            masked_spectrogram[f0:f1, :] = 0
        else:
            masked_spectrogram[f0:f1, :] = np.random.normal(0, 1, (f1 - f0, spectrogram.shape[1]))
    return masked_spectrogram


def time_mask(spectrogram, T=15, num_masks=1, replace_with_zero=True):
    """Applies time mask on the spectrogram
    Args:
        spectrogram: PyTorch Tensor, input spectrogram
        T: int, maximum size of each mask
        num_masks: int, number of masks to apply
        replace_with_zero: bool, replace masked values with zeros or not
    Returns:
        masked_spectrogram: PyTorch Tensor, spectrogram with time masks applied
    """
    masked_spectrogram = spectrogram.clone()
    num_frames = spectrogram.shape[1]
    for i in range(num_masks):
        t = torch.randint(0, num_frames - T, (1,)).item()
        t0 = t
        t1 = t + T
        if replace_with_zero:
            masked_spectrogram[:, t0:t1] = 0
        else:
            masked_spectrogram[:, t0:t1] = torch.randn_like(spectrogram[:, t0:t1])
    return masked_spectrogram


def plot_spectrogram(spectrogram, fs=8000, title="Spectrogram"):
    """Plots the spectrogram
    Args:
        spectrogram: numpy array, input spectrogram
        fs: int, sample rate of the signal
        title: str, title of the plot
    """
    fig, ax = plt.subplots()
    ax.set_title(title)
    cax = ax.matshow(np.log(spectrogram), origin='lower', aspect='auto', cmap='inferno')
    fig.colorbar(cax)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Frequency bin")
    plt.show()

def add_noise(waveform):
    # Generate Gaussian noise
    mean = 0
    std = 0.02
    noise = torch.empty_like(waveform).normal_(mean=mean, std=std)
    # Add the noise to the audio signal
    noisy_waveform = waveform + noise
    return noisy_waveform 
    
def pitch_shifting(data):
    sr  = 22050
    bins_per_octave = 12
    pitch_pm = 2
    pitch_change =  pitch_pm * 2*(np.random.uniform())   
    transform = PitchShift(sample_rate=sr, n_steps=pitch_change, bins_per_octave=bins_per_octave)
    for idx, sample in enumerate(data):
        sample = sample.unsqueeze(0)
        data[idx] = transform(sample)
    return data

def time_stretching(data, rate_mean=1):
    input_length = len(data[0])
    stretch = TimeStretch()
    for idx, sample in enumerate(data):
        rand_rate = np.random.normal(rate_mean, 0.2)
        data[idx] = stretch(sample, rand_rate)

        if len(data[idx]) > input_length:
            data[idx] = data[idx][:input_length]
        else:
            data[idx] = np.pad(data, (0, max(0, input_length - len(data[idx]))), "constant")
    return data