import logging
from pathlib import Path
from typing import Optional


log = logging.getLogger(__name__)


# taken and adapted from https://github.com/blakeembrey/language-map/blob/master/languages.json
LANGUAGE_MAP = {
    # "agda": {
    #   "type": "programming",
    #   "color": "#315665",
    #   "extensions": {
    #     ".agda"
    #   },
    # },
    "c": {
        "type": "programming",
        "color": "#555555",
        "extensions": {".c", ".cats", ".h", ".idc"},
        "interpreters": ["tcc"],
    },
    "cpp": {
        "type": "programming",
        "aliases": ["c++", "C++"],
        "color": "#f34b7d",
        "extensions": {
            ".cpp",
            ".c++",
            ".cc",
            ".cp",
            ".cxx",
            ".h",
            ".h++",
            ".hh",
            ".hpp",
            ".hxx",
            ".inc",
            ".inl",
            ".ino",
            ".ipp",
            ".re",
            ".tcc",
            ".tpp",
        },
    },
    "css": {
        "type": "markup",
        "color": "#563d7c",
        "extensions": {".css"},
    },
    "c-sharp": {
        "type": "programming",
        "color": "#178600",
        "aliases": ["csharp", "c#", "C#"],
        "extensions": {".cs", ".cake", ".cshtml", ".csx"},
    },
    "haskell": {
        "type": "programming",
        "color": "#5e5086",
        "extensions": {".hs", ".hsc"},
        "interpreters": ["runhaskell"],
    },
    "html": {
        "type": "markup",
        "color": "#e34c26",
        "aliases": ["xhtml"],
        "extensions": {".html", ".htm", ".html.hl", ".inc", ".st", ".xht", ".xhtml"},
    },
    "go": {
        "type": "programming",
        "color": "#375eab",
        "aliases": ["golang"],
        "extensions": {".go"},
    },
    "java": {
        "type": "programming",
        "color": "#b07219",
        "extensions": {".java"},
    },
    "javascript": {
        "type": "programming",
        "color": "#f1e05a",
        "aliases": ["js", "node"],
        "extensions": {
            ".js",
            "._js",
            ".bones",
            ".es",
            ".es6",
            ".frag",
            ".gs",
            ".jake",
            ".jsb",
            ".jscad",
            ".jsfl",
            ".jsm",
            ".jss",
            ".mjs",
            ".njs",
            ".pac",
            ".sjs",
            ".ssjs",
            ".xsjs",
            ".xsjslib",
        },
        "interpreters": ["node"],
    },
    # "json": {
    #   "type": "data",
    #   "extensions": {
    #     ".json",
    #     ".avsc",
    #     ".geojson",
    #     ".gltf",
    #     ".JSON-tmLanguage",
    #     ".jsonl",
    #     ".tfstate",
    #     ".tfstate.backup",
    #     ".topojson",
    #     ".webapp",
    #     ".webmanifest",
    #     ".yy",
    #     ".yyp"
    #   },
    # },
    "julia": {
        "type": "programming",
        "color": "#a270ba",
        "extensions": {".jl"},
        "interpreters": ["julia"],
    },
    "ocaml": {
        "type": "programming",
        "color": "#3be133",
        "extensions": {".ml", ".eliom", ".eliomi", ".ml4", ".mli", ".mll", ".mly"},
        "interpreters": ["ocaml", "ocamlrun", "ocamlscript"],
    },
    "php": {
        "type": "programming",
        "color": "#4F5D95",
        "extensions": {
            ".php",
            ".aw",
            ".ctp",
            ".fcgi",
            ".inc",
            ".php3",
            ".php4",
            ".php5",
            ".phps",
            ".phpt",
        },
        "interpreters": ["php"],
    },
    "python": {
        "color": "#3572A5",
        "extensions": {
            ".py",
            ".bzl",
            ".cgi",
            ".fcgi",
            ".gyp",
            ".gypi",
            ".lmi",
            ".py3",
            ".pyde",
            ".pyi",
            ".pyp",
            ".pyt",
            ".pyw",
            ".rpy",
            ".spec",
            ".tac",
            ".wsgi",
            ".xpy",
        },
        "interpreters": ["python", "python2", "python3"],
        "aliases": ["rusthon", "python3", "python2"],
    },
    "ruby": {
        "type": "programming",
        "color": "#701516",
        "aliases": ["jruby", "macruby", "rake", "rb", "rbx"],
        "extensions": {
            ".rb",
            ".builder",
            ".eye",
            ".fcgi",
            ".gemspec",
            ".god",
            ".jbuilder",
            ".mspec",
            ".pluginspec",
            ".podspec",
            ".rabl",
            ".rake",
            ".rbuild",
            ".rbw",
            ".rbx",
            ".ru",
            ".ruby",
            ".spec",
            ".thor",
            ".watchr",
        },
        "interpreters": ["ruby", "macruby", "rake", "jruby", "rbx"],
    },
    "rust": {
        "type": "programming",
        "color": "#dea584",
        "extensions": {".rs", ".rs.in"},
    },
    "scala": {
        "type": "programming",
        "color": "#c22d40",
        "extensions": {".scala", ".kojo", ".sbt", ".sc"},
        "interpreters": ["scala"],
    },
    # "swift": {
    #   "type": "programming",
    #   "color": "#ffac45",
    #   "extensions": {
    #     ".swift"
    #   },
    # },
    "typescript": {
        "type": "programming",
        "color": "#2b7489",
        "aliases": ["ts"],
        "interpreters": ["node"],
        "extensions": {".ts", ".tsx"},
    },
    # "verilog": {
    #   "type": "programming",
    #   "color": "#b2b7f8",
    #   "extensions": {
    #     ".v",
    #     ".veo"
    #   },
    # },
}


def languages_for_extension(
    language_map: dict[str, dict[str, str]]
) -> dict[str, list[str]]:
    extension_map = {}
    for language, data in language_map.items():
        for extension in data["extensions"]:
            if extension not in extension_map:
                extension_map[extension] = []

            extension_map[extension].append(language)

    return extension_map


EXTENSION_MAP = languages_for_extension(LANGUAGE_MAP)
ALIAS_TO_LANGUAGE = {
    alias: lang
    for lang, lang_data in LANGUAGE_MAP.items()
    for alias in lang_data.get("aliases", []) + [lang]
}


def detect_lang(filepath: Path) -> Optional[str]:
    possible_langs = detect_langs(filepath)

    if possible_langs is None:
        return

    # just return the one we found
    if len(possible_langs) == 1:
        return possible_langs[0]
    else:
        log.info(f"{filepath}  belongs to multiple languages: {possible_langs}.")


def detect_langs(filepath: Path) -> Optional[list[str]]:
    extension = filepath.suffix

    try:
        possible_langs = EXTENSION_MAP[extension]
    except KeyError:
        log.debug(f"{filepath} has unknown extension '{extension}'.")
        return

    # just return the one we found
    if len(possible_langs) == 1:
        return possible_langs

    # check if there exists a file with same name, but different suffix
    points = {lang: 0 for lang in possible_langs}
    for lang in possible_langs:
        for ext in LANGUAGE_MAP[lang]["extensions"]:
            if ext != extension and filepath.with_suffix(ext).exists():
                points[lang] += 1

    langs_with_max_score = [
        lang for lang, score in points.items() if score == max(points.values())
    ]

    return langs_with_max_score


def get_language_identifier(language_name: str) -> Optional[str]:
    """returns the identifier for this language from an arbitrary string"""
    if not isinstance(language_name, str):
        return

    if language_name in ALIAS_TO_LANGUAGE:
        return ALIAS_TO_LANGUAGE[language_name]
    elif language_name.lower() in ALIAS_TO_LANGUAGE:
        return ALIAS_TO_LANGUAGE[language_name.lower()]
    else:
        log.warning(f"Language identifier {language_name} unknown.")
        return language_name
