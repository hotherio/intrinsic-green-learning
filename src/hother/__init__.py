try:
    from importlib.metadata import version

    __version__ = version("hother-change-me")
except Exception:
    __version__ = "0.0.0"
