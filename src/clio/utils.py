import json
import pandas as pd
from global_variables import topics

def split_str(string_, char=','):
    """Split string on a character."""
    return string_.split(char)

def flatten_lists(lst):
    """Remove nested lists."""
    return [item for sublist in lst for item in sublist]

def read_json(path_to_json):
    """Read JSON file."""
    with open(path_to_json) as f:
        json_file = json.load(f)

    return json_file

def json2df(json_file, topics=topics):
    """Parse JSON file to Pandas DataFrame."""
    frames = []
    for key in topics:
        frame = pd.DataFrame(json_file[key])
        frame['Topic'] = key
        frames.append(frame)

    data = pd.concat(frames)
    data.drop_duplicates('id', inplace=True)
    data.reset_index(inplace=True, drop=True)
    data['id'] = ['_'.join(['id', str(i)]) for i in range(0, data.shape[0])]

    return data

def sample_topics(data, topic, n=3500, replace=False):
    """Sample a dataframe based on a topic.

    Args:
        data (Pandas Dataframe): It must contain a 'Topic' column.
        topic (str): MAK topic to subset the data on.
        n (int): Number of rows to sample.

    Returns:
        Subset of the data.

    """
    return data[data['Topic'] == topic].sample(n=n, replace=replace, random_state=42)

def make_subset(data, topics=topics, n=3500, replace=False):
    """Concatenate data samples.

    Args:
        df (Pandas DataFrame): It must contain a 'Topic' column.
        topics (list, str): MAK topics.

    Returns:
        Pandas DataFrame.

    """
    return pd.concat([sample_topics(data, topic, n, replace) for topic in topics
                        if topic in data['Topic'].unique()])

def unique_vals(d):
    """Find the unique tags in of a collection.

    Args:
        d (dict): Python dictionary.

    Returns:
        A set of the dictionary's unique values.

    """
    val_lst = [v for k, v in d.items()]
    return set(flatten_lists(val_lst))

def keep_row(relevant_tags, non_relevant_tags, val):
    """Keep only DF rows that contain items from one set and not anything else."""
    if any(tag in val for tag in relevant_tags) and not any(tag in val for tag in non_relevant_tags):
        return 1
    else:
        return 0
