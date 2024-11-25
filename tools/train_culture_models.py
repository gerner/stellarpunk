"""
Trains markov models for each culture.

These are saved in some location and can be used by stellarpunk to generate 
place, people, ship names.
"""

import sys
import logging
import gzip
import os.path
import time

import numpy as np

from stellarpunk import util
from stellarpunk.generate import markov

logger = logging.getLogger(__name__)

CULTURES = [
    #"balkan",
    #"caribbean",
    #"eastafrica",
    #"eastasia",
    #"hispania",
    #"mideast",
    #"northafrica",
    #"northamerica",
    "oceana",
    "scandinavia",
    "slavic",
    "southafrica",
    "southamerica",
    "southasia",
    "southeastasia",
    "westafrica",
    "westeurope",
]

SHIP_NAME_FILE = "/tmp/shipnames.txt"
PEOPLE_NAME_DIR = "/tmp/peoplenames"
PLACE_NAME_DIR = "/tmp/geonames"
MODELS_DIR = "/tmp/stellarpunk_models"

def train_save_model(m:markov.MarkovModel, input_filename, output_filename) -> None:
    if input_filename.endswith(".gz"):
        in_f = gzip.open(input_filename, "rt", errors="ignore")
    else:
        in_f = open(input_filename, "rt", errors="ignore")
    m.train(in_f)
    in_f.close()
    if output_filename.endswith(".gz"):
        out_f = gzip.open(output_filename, "wb")
    m.save(out_f)
    out_f.close()

    return m

def main() -> None:
    logging.basicConfig(
            format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
            stream=sys.stderr,
            level=logging.INFO
    )

    r = np.random.default_rng()

    logger.info("training ship name model")

    # model for ship names
    m = markov.MarkovModel(n=6, romanize=True, titleize=True, roman_numerals=True)
    train_save_model(m, SHIP_NAME_FILE, os.path.join(MODELS_DIR, "shipnames.mmodel.gz"))
    logger.info(f'sample ship: "{m.generate(r)}"')

    culture_start_time = time.time()
    for i, culture in enumerate(CULTURES):
        elapsed_time = time.time()-culture_start_time
        estimated_time = elapsed_time / i * len(CULTURES) if i > 0 else 3825
        logger.info(f'elapsed {util.human_timespan(elapsed_time)} est {util.human_timespan(estimated_time - elapsed_time)} remain. {float(i)/float(len(CULTURES))*100.0:.0f}%')
        logger.info(f'training models for {culture}')
        # models: first name, last name, sector name, station name

        # first and last names
        m = markov.MarkovModel(n=6, romanize=True, titleize=True, roman_numerals=False)
        logger.info(f'first names')
        train_save_model(m, os.path.join(PEOPLE_NAME_DIR, f'firstnames.{culture}.gz'), os.path.join(MODELS_DIR, f'firstnames.{culture}.mmodel.gz'))
        sample_first = m.generate(r)
        logger.info(f'last names')
        train_save_model(m, os.path.join(PEOPLE_NAME_DIR, f'lastnames.{culture}.gz'), os.path.join(MODELS_DIR, f'lastnames.{culture}.mmodel.gz'))
        sample_last = m.generate(r)
        logger.info(f'sample name: "{sample_first} {sample_last}"')

        # sector names
        m = markov.MarkovModel(n=6, romanize=True, titleize=True, roman_numerals=False)
        logger.info('sector names')
        train_save_model(m, os.path.join(PLACE_NAME_DIR, f'features.{culture}.gz'), os.path.join(MODELS_DIR, f'sectors.{culture}.mmodel.gz'))
        logger.info(f'sample sector: "{m.generate(r)}"')

        # station names
        m = markov.MarkovModel(n=6, romanize=True, titleize=True, roman_numerals=False)
        logger.info('station names')
        train_save_model(m, os.path.join(PLACE_NAME_DIR, f'adminareas.{culture}.gz'), os.path.join(MODELS_DIR, f'stations.{culture}.mmodel.gz'))
        logger.info(f'sample station: "{m.generate(r)}"')

    logger.info(f'done!')

if __name__ == "__main__":
    main()
