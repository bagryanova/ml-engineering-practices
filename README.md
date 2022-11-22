# ml-engineering-practices

Trains different classifiers for voice to male/female classes.

I chose package manager `pip`. Installation guide can be found here https://pip.pypa.io/en/stable/installation/

Tested on OS-X and python 3.9.7

Build the package:
1. activate venv: `python3 -m venv .venv && source .venv/bin/activate`
2. install dev requirements: `pip3 install -r requirements_dev.txt`
3. Run `python3 setup.py sdist`

Publish to test.pypi:
1. Run `twine upload --repository-url https://test.pypi.org/legacy/ dist/*`

Package on pypi: `https://test.pypi.org/project/sound-classifiers/0.0.6/`

Install from pypi:
`pip3 install -i https://test.pypi.org/simple/ sound-classifiers==0.0.6 && pip3 install -r requirements.txt`

Run example:
```
from sound_classifiers.main import main

main()
```
