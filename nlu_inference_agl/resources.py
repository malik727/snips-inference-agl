from __future__ import unicode_literals

import json
from copy import deepcopy
from pathlib import Path

from nlu_inference_agl.common.utils import get_package_path, is_package
from nlu_inference_agl.constants import (
    CUSTOM_ENTITY_PARSER_USAGE, DATA_PATH, GAZETTEERS, NOISE,
    STEMS, STOP_WORDS, WORD_CLUSTERS, METADATA)
from nlu_inference_agl.entity_parser.custom_entity_parser import (
    CustomEntityParserUsage)


class MissingResource(LookupError):
    pass


def load_resources(name, required_resources=None):
    """Load language specific resources

    Args:
        name (str): Resource name as in ``snips-nlu download <name>``. Can also
            be the name of a python package or a directory path.
        required_resources (dict, optional): Resources requirement
            dict which, when provided, allows to limit the amount of resources
            to load. By default, all existing resources are loaded.
    """
    if name in set(d.name for d in DATA_PATH.iterdir()):
        return load_resources_from_dir(DATA_PATH / name, required_resources)
    elif is_package(name):
        package_path = get_package_path(name)
        resources_sub_dir = get_resources_sub_directory(package_path)
        return load_resources_from_dir(resources_sub_dir, required_resources)
    elif Path(name).exists():
        path = Path(name)
        if (path / "__init__.py").exists():
            path = get_resources_sub_directory(path)
        return load_resources_from_dir(path, required_resources)
    else:
        raise MissingResource("Language resource '{r}' not found. This may be "
                              "solved by running "
                              "'python -m snips_nlu download {r}'"
                              .format(r=name))


def load_resources_from_dir(resources_dir, required_resources=None):
    with (resources_dir / "metadata.json").open(encoding="utf8") as f:
        metadata = json.load(f)
    metadata = _update_metadata(metadata, required_resources)
    gazetteer_names = metadata["gazetteers"]
    clusters_names = metadata["word_clusters"]
    stop_words_filename = metadata["stop_words"]
    stems_filename = metadata["stems"]
    noise_filename = metadata["noise"]

    gazetteers = _get_gazetteers(resources_dir / "gazetteers", gazetteer_names)
    word_clusters = _get_word_clusters(resources_dir / "word_clusters",
                                       clusters_names)

    stems = None
    stop_words = None
    noise = None

    if stems_filename is not None:
        stems = _get_stems(resources_dir / "stemming", stems_filename)
    if stop_words_filename is not None:
        stop_words = _get_stop_words(resources_dir, stop_words_filename)
    if noise_filename is not None:
        noise = _get_noise(resources_dir, noise_filename)

    return {
        METADATA: metadata,
        WORD_CLUSTERS: word_clusters,
        GAZETTEERS: gazetteers,
        STOP_WORDS: stop_words,
        NOISE: noise,
        STEMS: stems,
    }


def _update_metadata(metadata, required_resources):
    if required_resources is None:
        return metadata
    metadata = deepcopy(metadata)
    required_gazetteers = required_resources.get(GAZETTEERS, [])
    required_word_clusters = required_resources.get(WORD_CLUSTERS, [])
    for gazetter in required_gazetteers:
        if gazetter not in metadata["gazetteers"]:
            raise ValueError("Unknown gazetteer for language '%s': '%s'"
                             % (metadata["language"], gazetter))
    for word_clusters in required_word_clusters:
        if word_clusters not in metadata["word_clusters"]:
            raise ValueError("Unknown word clusters for language '%s': '%s'"
                             % (metadata["language"], word_clusters))
    metadata["gazetteers"] = required_gazetteers
    metadata["word_clusters"] = required_word_clusters
    if not required_resources.get(STEMS, False):
        metadata["stems"] = None
    if not required_resources.get(NOISE, False):
        metadata["noise"] = None
    if not required_resources.get(STOP_WORDS, False):
        metadata["stop_words"] = None
    return metadata


def get_resources_sub_directory(resources_dir):
    resources_dir = Path(resources_dir)
    with (resources_dir / "metadata.json").open(encoding="utf8") as f:
        metadata = json.load(f)
    resource_name = metadata["name"]
    version = metadata["version"]
    sub_dir_name = "{r}-{v}".format(r=resource_name, v=version)
    return resources_dir / sub_dir_name


def get_stop_words(resources):
    return _get_resource(resources, STOP_WORDS)


def get_noise(resources):
    return _get_resource(resources, NOISE)


def get_word_clusters(resources):
    return _get_resource(resources, WORD_CLUSTERS)


def get_word_cluster(resources, cluster_name):
    word_clusters = get_word_clusters(resources)
    if cluster_name not in word_clusters:
        raise MissingResource("Word cluster '{}' not found" % cluster_name)
    return word_clusters[cluster_name]


def get_gazetteer(resources, gazetteer_name):
    gazetteers = _get_resource(resources, GAZETTEERS)
    if gazetteer_name not in gazetteers:
        raise MissingResource("Gazetteer '%s' not found in resources"
                              % gazetteer_name)
    return gazetteers[gazetteer_name]


def get_stems(resources):
    return _get_resource(resources, STEMS)


def merge_required_resources(lhs, rhs):
    if not lhs:
        return dict() if rhs is None else rhs
    if not rhs:
        return dict() if lhs is None else lhs
    merged_resources = dict()
    if lhs.get(NOISE, False) or rhs.get(NOISE, False):
        merged_resources[NOISE] = True
    if lhs.get(STOP_WORDS, False) or rhs.get(STOP_WORDS, False):
        merged_resources[STOP_WORDS] = True
    if lhs.get(STEMS, False) or rhs.get(STEMS, False):
        merged_resources[STEMS] = True
    lhs_parser_usage = lhs.get(CUSTOM_ENTITY_PARSER_USAGE)
    rhs_parser_usage = rhs.get(CUSTOM_ENTITY_PARSER_USAGE)
    parser_usage = CustomEntityParserUsage.merge_usages(
        lhs_parser_usage, rhs_parser_usage)
    merged_resources[CUSTOM_ENTITY_PARSER_USAGE] = parser_usage
    gazetteers = lhs.get(GAZETTEERS, set()).union(rhs.get(GAZETTEERS, set()))
    if gazetteers:
        merged_resources[GAZETTEERS] = gazetteers
    word_clusters = lhs.get(WORD_CLUSTERS, set()).union(
        rhs.get(WORD_CLUSTERS, set()))
    if word_clusters:
        merged_resources[WORD_CLUSTERS] = word_clusters
    return merged_resources


def _get_resource(resources, resource_name):
    if resource_name not in resources or resources[resource_name] is None:
        raise MissingResource("Resource '%s' not found" % resource_name)
    return resources[resource_name]


def _get_stop_words(resources_dir, stop_words_filename):
    if not stop_words_filename:
        return None
    stop_words_path = (resources_dir / stop_words_filename).with_suffix(".txt")
    return _load_stop_words(stop_words_path)


def _load_stop_words(stop_words_path):
    with stop_words_path.open(encoding="utf8") as f:
        stop_words = set(l.strip() for l in f if l)
    return stop_words


def _get_noise(resources_dir, noise_filename):
    if not noise_filename:
        return None
    noise_path = (resources_dir / noise_filename).with_suffix(".txt")
    return _load_noise(noise_path)


def _load_noise(noise_path):
    with noise_path.open(encoding="utf8") as f:
        # Here we split on a " " knowing that it's always ignored by
        # the tokenization (see tokenization unit tests)
        # It is not important to tokenize precisely as this noise is just used
        # to generate utterances for the None intent
        noise = [word for l in f for word in l.split()]
    return noise


def _get_word_clusters(word_clusters_dir, clusters_names):
    if not clusters_names:
        return dict()

    clusters = dict()
    for clusters_name in clusters_names:
        clusters_path = (word_clusters_dir / clusters_name).with_suffix(".txt")
        clusters[clusters_name] = _load_word_clusters(clusters_path)
    return clusters


def _load_word_clusters(path):
    clusters = dict()
    with path.open(encoding="utf8") as f:
        for line in f:
            split = line.rstrip().split("\t")
            if not split:
                continue
            clusters[split[0]] = split[1]
    return clusters


def _get_gazetteers(gazetteers_dir, gazetteer_names):
    if not gazetteer_names:
        return dict()

    gazetteers = dict()
    for gazetteer_name in gazetteer_names:
        gazetteer_path = (gazetteers_dir / gazetteer_name).with_suffix(".txt")
        gazetteers[gazetteer_name] = _load_gazetteer(gazetteer_path)
    return gazetteers


def _load_gazetteer(path):
    with path.open(encoding="utf8") as f:
        gazetteer = set(v.strip() for v in f if v)
    return gazetteer



def _get_stems(stems_dir, filename):
    if not filename:
        return None
    stems_path = (stems_dir / filename).with_suffix(".txt")
    return _load_stems(stems_path)


def _load_stems(path):
    with path.open(encoding="utf8") as f:
        stems = dict()
        for line in f:
            elements = line.strip().split(',')
            stem = elements[0]
            for value in elements[1:]:
                stems[value] = stem
    return stems

