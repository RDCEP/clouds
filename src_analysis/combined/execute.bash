#!/bin/bash

# -cut-dirs ; number of directory name from head of URL

echo ${1} ${2} ${3}
bname=`basename ${1}`
file=${3}/${bname}
echo `wget -e robots=off -m -np -R .html,.tmp -nH --cut-dirs=6 "${1}" --header "Authorization: Bearer ${2}" -P ${3} && touch  ${file}.is.done`

echo 'NORMAL END'
