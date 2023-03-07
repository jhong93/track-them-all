# One Repo to Track Them All

One Repo to rule them all, One Repo to find them, One Repo to track them all and in the viewer bind them.

## Installation

Clone recursively with all submodules.

First, use `pip3 install -r requirements.txt`.

Second, install ViTPose in `deps/ViTPose`. Run `pip install -v -e .`

Third, install EVA in `deps/EVA/det`. Run `pip install -e .`

Comment out the assertion in `deps/ViTPose/mmpose/__init__.py`

## Usage

Run models with `pipeline.py`. View and label with `viewer.py`.