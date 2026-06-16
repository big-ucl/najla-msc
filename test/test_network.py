"""
Module: test/test_network.py

Description:
    Unit tests for the LOCATION_REGEXES dictionary defined in activitygraphs.data.geneva.

    LOCATION_REGEXES maps each location type name to a regular expression that uniquely
    identifies that type of location string.  Tests verify:
      - Each regex matches all provided examples for its own type.
      - Each regex does NOT match examples from any other type.

    This guards against regressions in the location-classification logic that maps raw
    string location IDs from the Geneva survey data to their semantic category.
"""

import re

import pytest

from activitygraphs.data.geneva import LOCATION_REGEXES


@pytest.fixture
def location_examples():
    """
    Description:
        Pytest fixture providing a small set of example location ID strings for each
        recognised location type, plus some "other" strings that should not match any
        of the regexes being tested.

    Output:
      - (dict[str, list[str]]): maps each location type key to a list of example strings.
    """
    return {
        "subsector": ["Aéroport - Arena - sous_secteur", "Pierres-du-Niton - sous_secteur"],
        "municipality_swiss": ["Genève - 1203"],          # Swiss municipality: name + NPA
        "municipality_french": ["FERNEY-VOLTAIRE - 01210"],  # French municipality: name + code postal
        "na": ["NA"],                                      # not-assigned placeholder string
        "other": ["Genève-Cornavin"],                      # a PT stop — should not match any type
    }


@pytest.mark.parametrize("location_type", LOCATION_REGEXES.keys())
def test_location_regex_passes_own_example(location_type: str, location_examples: dict):
    """
    Description:
        Verifies that the regex for each location_type successfully matches all
        example strings defined for that type in the location_examples fixture.
        A regression here means a valid location ID would be misclassified as unknown.

    Input:
      - location_type (str): one of the keys of LOCATION_REGEXES (parametrised).
      - location_examples (dict): the fixture providing example strings per type.

    Assertion:
      - re.match(LOCATION_REGEXES[location_type], example) is truthy for every
        example string for this location_type.
    """
    for example in location_examples[location_type]:
        assert re.match(LOCATION_REGEXES[location_type], example)


@pytest.mark.parametrize("location_type", LOCATION_REGEXES.keys())
def test_location_regex_fails_other_examples(location_type: str, location_examples: dict[str, str]):
    """
    Description:
        Verifies that the regex for each location_type does NOT match any example
        strings that belong to a different location type or the "other" category.
        A regression here means the regex is too broad and would misclassify unrelated
        location strings (e.g. classifying a PT stop as a subsector).

    Input:
      - location_type (str): one of the keys of LOCATION_REGEXES (parametrised).
      - location_examples (dict): the fixture providing example strings per type.

    Assertion:
      - re.match(LOCATION_REGEXES[location_type], other_example) is falsy for every
        example string that belongs to a different location type.
    """
    regex = LOCATION_REGEXES[location_type]
    # All types except the one being tested (including "other" which should match none)
    other_location_types = set(location_examples.keys()) - {location_type}

    for other_type in other_location_types:
        for other_example in location_examples[other_type]:
            assert not re.match(regex, other_example)
