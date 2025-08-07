#! /usr/bin/env bash

# Default values
LOCAL_ROOT=$(git rev-parse --show-toplevel)
cd "${LOCAL_ROOT}" || exit

source "leonardo/.env"
REMOTE_ROOT=${LEONARDO_USER}@${LEONARDO_DATA_MOVER}:${LEONARDO_ROOT}

if [[ ${1} = 'pull' ]]; then
  SOURCE="${REMOTE_ROOT}/"
  DESTINATION="./"
elif [[ ${1} = 'push' ]]; then
  SOURCE="./"
  DESTINATION="${REMOTE_ROOT}/"
else
  echo "First argument must be either 'push' or 'pull'"
  exit 1
fi

rsync -av "${SOURCE}" "${DESTINATION}" --exclude-from 'leonardo/.ignore' "${@:2}"