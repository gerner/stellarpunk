#!/bin/bash

###############################################################################
# using data from https://download.geonames.org/export/dump/
# Processes geonames all features file into a set of per-"culture" feature
# files.
###############################################################################

set -eu -o pipefail

country_file=$(mktemp)

cultures="$(cat countriescultures.tsv | tr -d '\r' | tail -n+2 | cut -f 4 | sort -u)"
for culture in $cultures; do
    echo ${culture}

    # natural features
    cat countriescultures.tsv | tr -d '\r' | awk -F\\t '$4 == "'${culture}'" { print $1 }' | LC_ALL=C sort > $country_file
    pv allfeatures.countrysorted.gz \
        | zcat \
        | awk -F\\t '($7=="U" || $7=="H" || $7=="T" || $7=="V")' \
        | LC_ALL=C join -t$'\t' -19  - $country_file \
        | cut -f 3 \
        | gzip -c \
        > features.${culture}.gz

    # admin areas: countries, states, cities, etc.
    pv allfeatures.countrysorted.gz \
        | zcat \
        | awk -F\\t '($7=="A" || $7=="P" || ($7=="S" && ($8=="AIRB" || $8=="AIRF" || $8=="AIRH" || $8=="AIRP" || $8=="AIRT" || $8=="ASTR" || $8=="CSTL" || $8=="CTRS" || $8=="EST" || $8=="INSM" || $8=="MN" || $8=="OILR" || $8=="STNB" || $8=="STNC" || $8=="STNE" || $8=="STNS" || $8=="TRANT")))' \
        | LC_ALL=C join -t$'\t' -19  - $country_file \
        | cut -f 3 \
        | gzip -c \
        > adminareas.${culture}.gz
done

