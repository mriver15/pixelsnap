"""pixelsnap: turn diffusion "pixel art" into real, verified pixel art. Run `pixelsnap --help`."""
try:
    from importlib.metadata import PackageNotFoundError, version as _version
    __version__ = _version("pixelsnap")
except Exception:  # not installed (running from a checkout)
    __version__ = "0.0.0+local"
