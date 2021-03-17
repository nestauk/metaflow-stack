"""Research DAPS."""
import os
from pathlib import Path

import sh


def _git_crypt_unlock():
    """Fetch git-crypt key from s3 and git-crypt unlock."""
    cwd = os.getcwd()
    os.chdir(Path(__file__).parent)
    try:
        sh.aws.s3.cp("s3://nesta-production-config/research_daps.key", ".")
        sh.git_crypt.unlock("research_daps.key")
    except sh.ErrorReturnCode:
        raise
    finally:
        sh.rm("research_daps.key")  # Cleanup key
        os.chdir(cwd)  # Restore working director


################################################################
# Text automatically added by daps-utils metaflowtask-init     #
from .__initplus__ import __basedir__, load_config  # noqa: E402 F401

try:
    config = load_config()
except ModuleNotFoundError as exc:
    print(exc)
except UnicodeDecodeError:
    _git_crypt_unlock()
    config = load_config()

################################################################


def declarative_base(prefix=""):
    """SQL alchemy declarative base.

    Converts table names from camel to snake case and `prefix`.

    Args:
        prefix: Table name prefix.

    Returns:
        Declarative base
    """
    from sqlalchemy.ext.declarative import declarative_base, declared_attr
    import re

    def camel_to_snake(str_):
        return re.sub(r"((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))", r"_\1", str_).lower()

    class _Base(object):
        """Research DAPS Base object."""

        @declared_attr
        def __tablename__(cls):  # noqa: B902 N805
            return prefix + camel_to_snake(cls.__name__)

    return declarative_base(cls=_Base)
