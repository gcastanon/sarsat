# ruff: noqa: E501
"""The natural-language request parser: rules, gazetteer and the pipeline around them."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live import nl
from live.nl.geocode import Gazetteer, offset_point
from live.nl.rules import parse_coordinates, parse_deadline, parse_priority

CITIES = [
    # geonameid name asciiname alternates lat lon fclass fcode cc cc2 adm1 adm2 adm3 adm4 pop
    (2747891, "Rotterdam", "Rotterdam", "Roterdam,Rotterdam", 51.9225, 4.47917, "P", "PPL", "NL", "", "11", "", "", "", 598199),
    (2988507, "Paris", "Paris", "Pari,Parigi,Paris", 48.85341, 2.3488, "P", "PPLC", "FR", "", "11", "", "", "", 2138551),
    (4717560, "Paris", "Paris", "Paris", 33.66094, -95.55551, "P", "PPL", "US", "", "TX", "", "", "", 24171),
    (2147714, "Sydney", "Sydney", "Sidney,Sydney", -33.86785, 151.20732, "P", "PPLA", "AU", "", "02", "", "", "", 4627345),
    (1701668, "Manila", "Manila", "Manila,Maynila", 14.6042, 120.9822, "P", "PPLC", "PH", "", "NCR", "", "", "", 1600000),
    (5392171, "San Jose", "San Jose", "San Jose", 37.33939, -121.89496, "P", "PPL", "US", "", "CA", "", "", "", 1026908),
    (2643743, "London", "London", "Londres,London", 51.50853, -0.12574, "P", "PPLC", "GB", "", "ENG", "", "", "", 8961989),
    (2510911, "Sevilla", "Sevilla", "Seville,Sevilla", 37.38283, -5.97317, "P", "PPLA", "ES", "", "51", "", "", "", 703206),
]  # fmt: skip
COUNTRIES = [
    ("NL", "NLD", "528", "NL", "Netherlands", "Amsterdam", "41526", "17231017", "EU"),
    ("FR", "FRA", "250", "FR", "France", "Paris", "547030", "66987244", "EU"),
    ("AU", "AUS", "036", "AS", "Australia", "Canberra", "7686850", "24992369", "OC"),
    ("US", "USA", "840", "US", "United States", "Washington", "9629091", "327167434", "NA"),
]
ADMIN1 = [
    ("US.CA", "California", "California", 5332921),
    ("AU.02", "New South Wales", "New South Wales", 2155400),
]


@pytest.fixture(scope="module")
def gaz(tmp_path_factory):
    d = tmp_path_factory.mktemp("geonames")
    with open(d / "cities15000.txt", "w", encoding="utf-8") as f:
        for row in CITIES:
            f.write("\t".join(str(x) for x in row) + "\t0\t0\tEurope/Amsterdam\t2024-01-01\n")
    with open(d / "countryInfo.txt", "w", encoding="utf-8") as f:
        f.write("# comment\n")
        for row in COUNTRIES:
            f.write("\t".join(row) + "\t.nl\tEUR\tEuro\t31\t####\t^\tnl\t2750405\tBE,DE\t\n")
    with open(d / "admin1CodesASCII.txt", "w", encoding="utf-8") as f:
        for row in ADMIN1:
            f.write("\t".join(str(x) for x in row) + "\n")
    return Gazetteer(str(d))


def test_rules_coordinates():
    assert parse_coordinates("image 48.86, 2.35 now") == (48.86, 2.35)
    assert parse_coordinates("lat 33.9 S lon 151.2 E") == (-33.9, 151.2)
    assert parse_coordinates("40.7N 74.0W please") == (40.7, -74.0)
    assert parse_coordinates("12.5,-70.0") == (12.5, -70.0)
    assert parse_coordinates("within 20 minutes, 3 sats") is None
    assert parse_coordinates("95, 10") is None  # out of range


def test_rules_deadline_and_priority():
    assert parse_deadline("within 20 minutes") == 20
    assert parse_deadline("in the next 5 min") == 5
    assert parse_deadline("10 minute window") == 10
    assert parse_deadline("no later than 25 mins") == 25
    assert parse_deadline("in 2 hours") == 120
    assert parse_deadline("within half an hour") == 30
    assert parse_deadline("asap") == 5
    assert parse_deadline("image Paris") is None
    assert parse_priority("urgent request") == "high"
    assert parse_priority("low priority") == "low"
    assert parse_priority("when convenient") == "low"
    assert parse_priority("standard priority") == "normal"
    assert parse_priority("image Paris") is None


def test_rules_pipeline_without_gazetteer():
    p = nl.parse("collect 40.7N 74.0W in 2 hours")
    assert (p.lat, p.lon) == (40.7, -74.0)
    assert p.deadline_minutes == 30 and "capped" in p.notes[0]
    p = nl.parse("image Rotterdam")
    assert not p.located and "no location" in p.notes[-1]


def test_gazetteer_lookup(gaz):
    hit = gaz.lookup("Rotterdam")
    assert hit and (round(hit.lat, 2), round(hit.lon, 2)) == (51.92, 4.48)
    assert gaz.lookup("port of Rotterdam").name == "Rotterdam"
    assert gaz.lookup("Roterdam").name == "Rotterdam"  # alternate name
    assert gaz.lookup("Paris").country == "France"  # the bigger Paris wins
    assert gaz.lookup("Paris, United States").country == "United States"
    assert gaz.lookup("Seville").name == "Sevilla"
    assert gaz.lookup("Rotterdamm").name == "Rotterdam"  # fuzzy
    assert gaz.lookup("Netherlands") is None  # its capital is not in this fixture
    assert gaz.lookup("France").name == "Paris, France"
    assert gaz.lookup("California").kind == "admin1" and "San Jose" in gaz.lookup("California").name
    assert gaz.lookup("Atlantis") is None


def test_gazetteer_find_in_text(gaz):
    place, offset = gaz.find_in_text(
        "image the port of Rotterdam, high priority, within 20 minutes"
    )
    assert place.name == "Rotterdam" and offset is None
    place, _ = gaz.find_in_text("urgent: SAR pass over San Jose in 10 min")
    assert place.name == "San Jose"
    place, offset = gaz.find_in_text("collect 50 km east of Manila within 15 minutes")
    assert place.name == "Manila" and offset == (50.0, "east")
    place, _ = gaz.find_in_text("please image Sydney now")
    assert place.name == "Sydney"


def test_offset_point():
    lat, lon = offset_point(0.0, 0.0, 111.19, "north")
    assert abs(lat - 1.0) < 1e-3 and abs(lon) < 1e-6
    lat, lon = offset_point(51.9, 4.5, 50, "east")
    assert abs(lat - 51.9) < 0.01 and lon > 4.5


class _FakeModel:
    def __init__(self, out):
        self.out = out

    def extract(self, text):
        return self.out


def test_pipeline_with_gazetteer_and_model(gaz):
    p = nl.parse("image the port of Rotterdam, high priority, within 20 minutes", geocoder=gaz)
    assert p.located and p.place == "Rotterdam" and p.source == "gazetteer"
    assert p.priority == "high" and p.deadline_minutes == 20
    model = _FakeModel(
        dict(
            place="Manila",
            lat=14.6,
            lon=121.0,
            offset_km=50.0,
            bearing="east",
            priority="low",
            deadline_minutes=12,
        )
    )
    p = nl.parse("fifty km east of Manila, routine, twelve minutes", geocoder=gaz, model=model)
    assert (
        p.source == "gazetteer"
        and p.lon > 121.0
        and p.priority == "low"
        and p.deadline_minutes == 12
    )
    model = _FakeModel(
        dict(
            place="Atlantis",
            lat=-30.0,
            lon=-40.0,
            offset_km=None,
            bearing=None,
            priority=None,
            deadline_minutes=None,
        )
    )
    p = nl.parse("image Atlantis", geocoder=gaz, model=model)
    assert (
        p.source == "model"
        and (p.lat, p.lon) == (-30.0, -40.0)
        and any("approximate" in n for n in p.notes)
    )
    p = nl.parse("image 48.86, 2.35", geocoder=gaz, model=_FakeModel(dict(place="Nowhere")))
    assert p.source == "coordinates"
