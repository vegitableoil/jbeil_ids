# !/usr/bin/bash

# conda env export --name kairos --file environment.yml

git add *

if [ -z "$1" ]
then
    git commit -m 'auto release'
else
    git commit -m "$1"
fi

#git push -u origin master
git push origin main
