#!/bin/bash

###############################################################################
# using FB leak data from https://github.com/philipperemy/name-datase
# Processes people name data from the archive into a set of per-"culture"
# name file pairs, first and last.
###############################################################################

set -eu -o pipefail
set -x

cultures="$(cat countriescultures.tsv | tr -d '\r' | tail -n+2 | cut -f 4 | sort -u)"

for culture in $cultures; do
    echo ${culture}
    country_files=$( \
        LC_ALL=C join <( \
            unzip -l name_dataset.zip \
                | ag -o 'name_dataset/.*$' \
                | LC_ALL=C sort \
        ) <( \
            cat countriescultures.tsv \
                | tr -d '\r' \
                | ag ${culture} \
                | cut -f1 \
                | xargs -i{} echo name_dataset/data/{}.csv \
                | LC_ALL=C sort \
        ) \
        | tr '\n' ' ' \
    )
    if [ -n "${country_files}" ]; then
        echo ${country_files}
        unzip -qc name_dataset.zip ${country_files} | tr ',' '\t' | cut -f 1 \
            | gzip -c > firstnames.${culture}.gz
        unzip -qc name_dataset.zip ${country_files} | tr ',' '\t' | cut -f 2 \
            | gzip -c > lastnames.${culture}.gz
    else
        echo "no matching files"
    fi
done

# this dataset does not represent all countries. see the output of this
# command for details:
#  LC_ALL=C join -v1 <(cat countriescultures.tsv | cut -f 1 | sed 's/.*/name_dataset\/data\/&.csv/' | LC_ALL=C sort) <(unzip -l name_dataset.zip | ag -o 'name_dataset/.*' | LC_ALL=C sort) | ag -o '[A-Z]*.csv$' | ag -o '^[A-Z]*' | LC_ALL=C join -t$'\t' - <(cat countriescultures.tsv | LC_ALL=C sort)
# in particular, only fiji exists in oceana. so we'll augment that with names
# from southeastasia and a small sample from westeurope

echo "augmenting oceana"

# we'll get this many from southeast asia and oceana
nearby_count=$(zcat firstnames.southeastasia.gz firstnames.oceana.gz | wc -l)
# we'll augment with this many from western europe
# ethnic breakdown of oceana includes a lot of people of european descent
# this oceana dataset does not include Australia, Papua New Guinea or New
# Zealand, the largest three countries in oceana that account for > 90%
# AU + NZ add in about 18M people of European descent which accounts for about
# 40% of the population of Oceana.
european_count=$(echo "${nearby_count} * 0.7" | bc -l | ag -o "^[0-9]*")

echo "${nearby_count} + ${european_count} = $((${nearby_count} + ${european_count})) target names"

echo "first names"
zcat firstnames.westeurope.gz | shuf -n $european_count | cat - <(zcat firstnames.southeastasia.gz firstnames.oceana.gz) | gzip -c > augmented_firstnames.oceana.gz
echo "last names"
zcat lastnames.westeurope.gz | shuf -n $european_count | cat - <(zcat lastnames.southeastasia.gz lasttnames.oceana.gz) | gzip -c > augmented_lastnames.oceana.gz

mv firstnames.oceana.gz original_firstnames.oceana.gz
mv lastnames.oceana.gz original_lastnames.oceana.gz
mv augmented_firstnames.oceana.gz firstnames.oceana.gz
mv augmented_lastnames.oceana.gz lastnames.oceana.gz
