#!/bin/bash

# -cut-dirs ; number of directory name from head of URL #6 = basename
# 1 - path
# 2 - appkey 
# 3 - where you save
echo 'START'

echo ${1} ${2} ${3}
echo `wget -e robots=off -m -np -R .html,.tmp -nH --cut-dirs=3 "${1}" --header "Authorization: Bearer ${2}" -P ${3}`

echo 'NORMAL END'
