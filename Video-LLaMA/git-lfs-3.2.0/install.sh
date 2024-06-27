#!/usr/bin/env bash
set -eu

prefix="/home/student.aau.dk/fd27ct/.local/bin"

if [ "${PREFIX:-}" != "" ] ; then
  prefix=${PREFIX:-}
elif [ "${BOXEN_HOME:-}" != "" ] ; then
  prefix=${BOXEN_HOME:-}
fi

mkdir -p $prefix/bin
rm -rf $prefix/bin/git-lfs*

pushd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null
  for g in git*; do
    install $g "$prefix/bin/$g"
  done
popd > /dev/null

PATH+=:$prefix/bin
git lfs install
