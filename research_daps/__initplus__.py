"""Extra initialisation for finding and loading config files."""


def path_to_init(_file=__file__, cast_to_str=False):
    """Return the path to this file."""
    from pathlib import Path

    path = Path(_file).resolve().parent
    return str(path) if cast_to_str else path


def path_to_this(this):
    """Return the path to what "this" is."""
    return path_to_init() / this


def load(path):
    """Load a config file from the given path."""
    import yaml
    import json
    import configparser

    with open(path) as f:
        if path.suffix in (".yml", ".yaml"):
            return yaml.safe_load(f)
        if path.suffix == ".json":
            return json.load(f)
        if path.suffix in (".cnf", ".conf", ".config"):
            config = configparser.ConfigParser()
            config.read(path)
            return config
        if path.suffix in (".sh",):
            return path
        raise ValueError(f'Unknown config file type "{path.suffix}"')


def recursive_load(path_to_config):
    """Recursively load config files or return path if not config."""
    config = {}
    for child in path_to_config.iterdir():
        if child.is_dir():
            config[child.name] = recursive_load(child)
        elif child.suffix == "":
            config[child.name] = str(child)
        else:
            config[child.stem] = load(child)
    return config


def load_config():
    """Load all of the config files."""
    path_to_config = path_to_this("config")
    return recursive_load(path_to_config)


__basedir__ = path_to_init(cast_to_str=True)
