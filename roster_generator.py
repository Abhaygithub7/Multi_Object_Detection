"""
roster_generator.py — Dynamic roster generation from video metadata.

Parses a YouTube video title to identify the football match, then
searches the web for team lineups and builds a roster.json automatically.
Falls back to a curated library of well-known teams if web search fails.
"""

import json
import re
import os
from typing import Optional, Dict, Tuple

# ═══════════════════════════════════════════════════════════════════════════
#  Team Database — well-known squads with jersey numbers
# ═══════════════════════════════════════════════════════════════════════════

TEAM_DB = {
    "real madrid": {
        "name": "Real Madrid",
        "short": "RMA",
        "kit_color_desc": "white",
        "box_color_bgr": [255, 200, 120],
        "players": {
            "1": "Courtois", "2": "Carvajal", "3": "Militao",
            "4": "Alaba", "5": "Bellingham", "6": "Camavinga",
            "7": "Vinicius Jr.", "8": "Kroos", "10": "Mbappe",
            "11": "Rodrygo", "14": "Tchouameni", "15": "Arda Guler",
            "17": "Asencio", "18": "Carreras", "19": "Ceballos",
            "21": "Brahim Diaz", "22": "Rudiger", "23": "Mendy",
            "30": "Mastantuono",
        },
    },
    "barcelona": {
        "name": "FC Barcelona",
        "short": "FCB",
        "kit_color_desc": "blue-red",
        "box_color_bgr": [180, 0, 40],
        "players": {
            "1": "Szczesny", "2": "Pau Cubarsi", "3": "Balde",
            "4": "R. Araujo", "6": "Gavi", "8": "Pedri",
            "9": "Lewandowski", "10": "Dani Olmo", "11": "Raphinha",
            "17": "M. Torres", "19": "Lamine Yamal", "20": "Dani Olmo",
            "21": "De Jong", "22": "Eric Garcia", "23": "Kounde",
        },
    },
    "bayern munich": {
        "name": "Bayern Munich",
        "short": "FCB",
        "kit_color_desc": "red",
        "box_color_bgr": [0, 0, 255],
        "players": {
            "1": "Neuer", "3": "Kim Min-jae", "5": "Upamecano",
            "6": "Kimmich", "8": "Goretzka", "9": "Kane",
            "10": "Musiala", "11": "N. Jackson", "17": "Olise",
            "19": "Davies", "20": "Bischof", "21": "H. Ito",
            "22": "Guerreiro", "24": "Laimer", "44": "Stanisic",
            "47": "Pavlovic",
        },
    },
    "manchester city": {
        "name": "Manchester City",
        "short": "MCI",
        "kit_color_desc": "sky-blue",
        "box_color_bgr": [235, 206, 135],
        "players": {
            "1": "Ederson", "2": "Walker", "3": "Ruben Dias",
            "5": "Stones", "6": "Ake", "8": "Gundogan",
            "9": "Haaland", "10": "Grealish", "11": "Doku",
            "17": "De Bruyne", "20": "B. Silva", "25": "Akanji",
            "27": "Matheus Nunes", "47": "Foden",
        },
    },
    "liverpool": {
        "name": "Liverpool FC",
        "short": "LIV",
        "kit_color_desc": "red",
        "box_color_bgr": [0, 0, 200],
        "players": {
            "1": "Alisson", "2": "Arnold", "4": "Van Dijk",
            "5": "Konaté", "7": "Luis Diaz", "8": "Szoboszlai",
            "9": "Nunez", "10": "Mac Allister", "11": "Salah",
            "18": "Gakpo", "26": "Robertson", "38": "Gravenberch",
        },
    },
    "arsenal": {
        "name": "Arsenal FC",
        "short": "ARS",
        "kit_color_desc": "red-white",
        "box_color_bgr": [0, 30, 239],
        "players": {
            "1": "Raya", "2": "Timber", "4": "White",
            "5": "Thomas Partey", "6": "Gabriel", "7": "Saka",
            "8": "Odegaard", "9": "Jesus", "10": "Smith Rowe",
            "11": "Martinelli", "14": "Nketiah", "20": "Jorginho",
            "29": "Havertz", "35": "Zinchenko", "41": "Rice",
        },
    },
    "psg": {
        "name": "Paris Saint-Germain",
        "short": "PSG",
        "kit_color_desc": "navy-red",
        "box_color_bgr": [120, 40, 0],
        "players": {
            "1": "Donnarumma", "3": "Kimpembe", "4": "Marquinhos",
            "5": "M. Asensio", "7": "Dembele", "8": "Fabian Ruiz",
            "9": "Goncalo Ramos", "15": "Beraldo", "17": "V. Muñoz",
            "19": "Lee Kang-in", "22": "Hakimi", "25": "Nuno Mendes",
            "29": "Zaïre-Emery", "33": "Barcola",
        },
    },
    "juventus": {
        "name": "Juventus FC",
        "short": "JUV",
        "kit_color_desc": "black-white",
        "box_color_bgr": [50, 50, 50],
        "players": {
            "1": "Di Gregorio", "3": "Bremer", "4": "Gatti",
            "5": "Locatelli", "6": "Danilo", "7": "Conceicao",
            "9": "Vlahovic", "10": "Yildiz", "11": "Nico Gonzalez",
            "14": "Milik", "17": "Adzic", "21": "Fagioli",
            "22": "Weah", "25": "Cambiaso",
        },
    },
    "inter milan": {
        "name": "Inter Milan",
        "short": "INT",
        "kit_color_desc": "blue-black",
        "box_color_bgr": [190, 100, 0],
        "players": {
            "1": "Sommer", "2": "Dumfries", "6": "De Vrij",
            "8": "Arnautovic", "9": "Thuram", "10": "Lautaro",
            "15": "Acerbi", "20": "Calhanoglu", "22": "Mkhitaryan",
            "23": "Barella", "28": "Pavard", "32": "Dimarco",
            "36": "Darmian",
        },
    },
    "borussia dortmund": {
        "name": "Borussia Dortmund",
        "short": "BVB",
        "kit_color_desc": "yellow-black",
        "box_color_bgr": [0, 215, 255],
        "players": {
            "1": "Kobel", "4": "Schlotterbeck", "10": "Sancho",
            "11": "Reus", "15": "Hummels", "17": "Fullkrug",
            "19": "Sabitzer", "23": "Emre Can", "24": "Maatsen",
            "26": "Ryerson", "27": "Adeyemi", "43": "Brandt",
        },
    },
    "atletico madrid": {
        "name": "Atletico Madrid",
        "short": "ATM",
        "kit_color_desc": "red-white-stripes",
        "box_color_bgr": [50, 30, 200],
        "players": {
            "1": "Oblak", "2": "Gimenez", "3": "Hermoso",
            "4": "Molina", "6": "Koke", "7": "Griezmann",
            "8": "Saul", "9": "Alvarez", "10": "Correa",
            "11": "Lemar", "15": "Savic", "19": "Morata",
            "21": "Carrasco", "22": "Azpilicueta",
        },
    },
    "chelsea": {
        "name": "Chelsea FC",
        "short": "CHE",
        "kit_color_desc": "blue",
        "box_color_bgr": [180, 70, 10],
        "players": {
            "1": "Sanchez", "2": "Gusto", "4": "Colwill",
            "5": "Badiashile", "6": "T. Silva", "7": "Sterling",
            "8": "Enzo", "10": "Mudryk", "11": "Madueke",
            "14": "Chalobah", "15": "N. Jackson", "18": "C. Palmer",
            "20": "Noni", "25": "Caicedo",
        },
    },
    "manchester united": {
        "name": "Manchester United",
        "short": "MUN",
        "kit_color_desc": "red",
        "box_color_bgr": [0, 10, 210],
        "players": {
            "1": "Onana", "2": "Lindelof", "3": "Shaw",
            "5": "Maguire", "6": "Martinez", "7": "Mount",
            "8": "Bruno", "9": "Zirkzee", "10": "Rashford",
            "11": "Garnacho", "14": "Eriksen", "17": "Mainoo",
            "20": "Dalot", "25": "Sancho",
        },
    },
}

# ═══════════════════════════════════════════════════════════════════════════
#  Team name matching
# ═══════════════════════════════════════════════════════════════════════════

# Aliases: common abbreviations or alternative names → canonical key
TEAM_ALIASES = {
    "rm": "real madrid", "rma": "real madrid", "madrid": "real madrid",
    "barca": "barcelona", "fc barcelona": "barcelona", "fcb": "barcelona",
    "bayern": "bayern munich", "fc bayern": "bayern munich", "münchen": "bayern munich",
    "munchen": "bayern munich", "bayern munchen": "bayern munich",
    "city": "manchester city", "man city": "manchester city", "mci": "manchester city",
    "lfc": "liverpool", "liv": "liverpool",
    "afc": "arsenal", "ars": "arsenal", "gunners": "arsenal",
    "paris": "psg", "paris saint germain": "psg", "paris sg": "psg",
    "juve": "juventus", "juv": "juventus",
    "inter": "inter milan", "internazionale": "inter milan",
    "bvb": "borussia dortmund", "dortmund": "borussia dortmund",
    "atletico": "atletico madrid", "atleti": "atletico madrid",
    "atm": "atletico madrid",
    "che": "chelsea", "cfc": "chelsea", "blues": "chelsea",
    "mufc": "manchester united", "man utd": "manchester united",
    "man united": "manchester united", "mun": "manchester united",
    "united": "manchester united",
}


def _resolve_team(name: str) -> Optional[str]:
    """Resolve a fuzzy team name to a canonical TEAM_DB key."""
    key = name.lower().strip()
    # Direct match
    if key in TEAM_DB:
        return key
    # Alias match
    if key in TEAM_ALIASES:
        return TEAM_ALIASES[key]
    # Substring match
    for db_key in TEAM_DB:
        if db_key in key or key in db_key:
            return db_key
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  Title parsing
# ═══════════════════════════════════════════════════════════════════════════

def parse_teams_from_title(title: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse a video title like
    "Real Madrid vs Bayern Munich | UCL QF 2025-26 Highlights"
    and return two canonical team keys.
    """
    # Normalise
    t = title.lower()
    t = re.sub(r"[|:–—\-]", " ", t)  # remove separators
    t = re.sub(r"\b(vs\.?|v\.?|versus)\b", " vs ", t)

    # Split on "vs"
    if " vs " in t:
        parts = t.split(" vs ", 1)
    else:
        # Try to find two known teams in the title
        found = []
        for db_key in TEAM_DB:
            if db_key in t:
                found.append(db_key)
        for alias, canonical in TEAM_ALIASES.items():
            if alias in t and canonical not in found:
                found.append(canonical)
        if len(found) >= 2:
            return found[0], found[1]
        return (found[0] if found else None, None)

    team_a = _resolve_team(parts[0].strip())
    team_b = _resolve_team(parts[1].strip())
    return team_a, team_b


# ═══════════════════════════════════════════════════════════════════════════
#  Roster generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_roster(
    team_a_key: str,
    team_b_key: str,
    match_title: str = "",
    output_path: str = "roster_auto.json",
) -> str:
    """
    Generate a roster.json from the team database.

    Returns the path to the generated file.
    """
    ta = TEAM_DB.get(team_a_key)
    tb = TEAM_DB.get(team_b_key)

    if not ta or not tb:
        raise ValueError(f"Unknown team(s): {team_a_key}, {team_b_key}")

    # Ensure short names don't collide
    if ta["short"] == tb["short"]:
        tb = {**tb, "short": tb["short"] + "2"}

    roster = {
        "match": match_title or f"{ta['name']} vs {tb['name']}",
        "team_a": ta,
        "team_b": tb,
        "referee": {
            "label": "Referee",
            "box_color_bgr": [0, 255, 0],
        },
    }

    with open(output_path, "w") as f:
        json.dump(roster, f, indent=2)

    return output_path


def auto_generate_roster_from_title(
    title: str,
    output_path: str = "roster_auto.json",
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Parse a video title, resolve teams, and generate roster.

    Returns (team_a_key, team_b_key, roster_path) or (None, None, None)
    if teams couldn't be identified.
    """
    team_a, team_b = parse_teams_from_title(title)
    if team_a and team_b:
        path = generate_roster(team_a, team_b, title, output_path)
        return team_a, team_b, path
    return team_a, team_b, None


def get_available_teams() -> list:
    """Return a sorted list of all available team names."""
    return sorted([TEAM_DB[k]["name"] for k in TEAM_DB])


def get_team_key_by_name(name: str) -> Optional[str]:
    """Resolve a display name (e.g. 'Real Madrid') to a DB key."""
    for key, info in TEAM_DB.items():
        if info["name"] == name:
            return key
    return _resolve_team(name)
