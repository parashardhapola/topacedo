rm -rf build/ dist/ topacedo.egg-info/
python -m build
twine upload --verbose dist/*
