# coding=utf-8
from __future__ import print_function, unicode_literals

import io
from itertools import cycle

from nlu_inference_agl.common.utils import unicode_string
from nlu_inference_agl.dataset.entity import Entity
from nlu_inference_agl.dataset.intent import Intent
from nlu_inference_agl.exceptions import DatasetFormatError


class Dataset(object):
    """Dataset used in the main NLU training API

    Consists of intents and entities data. This object can be built either from
    text files (:meth:`.Dataset.from_files`) or from YAML files
    (:meth:`.Dataset.from_yaml_files`).

    Attributes:
        language (str): language of the intents
        intents (list of :class:`.Intent`): intents data
        entities (list of :class:`.Entity`): entities data
    """

    def __init__(self, language, intents, entities):
        self.language = language
        self.intents = intents
        self.entities = entities
        self._add_missing_entities()
        self._ensure_entity_values()

    @classmethod
    def _load_dataset_parts(cls, stream, stream_description):
        from nlu_inference_agl.dataset.yaml_wrapper import yaml

        intents = []
        entities = []
        for doc in yaml.safe_load_all(stream):
            doc_type = doc.get("type")
            if doc_type == "entity":
                entities.append(Entity.from_yaml(doc))
            elif doc_type == "intent":
                intents.append(Intent.from_yaml(doc))
            else:
                raise DatasetFormatError(
                    "Invalid 'type' value in YAML file '%s': '%s'"
                    % (stream_description, doc_type))
        return intents, entities

    def _add_missing_entities(self):
        entity_names = set(e.name for e in self.entities)

        # Add entities appearing only in the intents utterances
        for intent in self.intents:
            for entity_name in intent.entities_names:
                if entity_name not in entity_names:
                    entity_names.add(entity_name)
                    self.entities.append(Entity(name=entity_name))

    def _ensure_entity_values(self):
        entities_values = {entity.name: self._get_entity_values(entity)
                           for entity in self.entities}
        for intent in self.intents:
            for utterance in intent.utterances:
                for chunk in utterance.slot_chunks:
                    if chunk.text is not None:
                        continue
                    try:
                        chunk.text = next(entities_values[chunk.entity])
                    except StopIteration:
                        raise DatasetFormatError(
                            "At least one entity value must be provided for "
                            "entity '%s'" % chunk.entity)
        return self

    def _get_entity_values(self, entity):
        from snips_nlu_parsers import get_builtin_entity_examples

        if entity.is_builtin:
            return cycle(get_builtin_entity_examples(
                entity.name, self.language))
        values = [v for utterance in entity.utterances
                  for v in utterance.variations]
        values_set = set(values)
        for intent in self.intents:
            for utterance in intent.utterances:
                for chunk in utterance.slot_chunks:
                    if not chunk.text or chunk.entity != entity.name:
                        continue
                    if chunk.text not in values_set:
                        values_set.add(chunk.text)
                        values.append(chunk.text)
        return cycle(values)

    @property
    def json(self):
        """Dataset data in json format"""
        intents = {intent_data.intent_name: intent_data.json
                   for intent_data in self.intents}
        entities = {entity.name: entity.json for entity in self.entities}
        return dict(language=self.language, intents=intents, entities=entities)
