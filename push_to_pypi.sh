file="build"
if [ -f "$file" ] ; then
    rm -r "$file"
fi
file="dist"
if [ -f "$file" ] ; then
    rm -r "$file"
fi
file="topacedo.egg-info"
if [ -f "$file" ] ; then
    rm -r "$file"
fi

python -m build
twine upload --verbose dist/*
