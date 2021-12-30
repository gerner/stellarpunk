import sys
import logging
import contextlib

import ipdb

from stellarpunk import util, interface, generate

class IPDBManager:
    def __init__(self):
        self.logger = logging.getLogger(util.fullname(self))

    def __enter__(self):
        self.logger.info("entering IPDBManager")

        return self

    def __exit__(self, e, m, tb):
        self.logger.info("exiting IPDBManager")
        if e is not None:
            self.logger.info(f'handling exception {e} {m}')
            print(m.__repr__(), file=sys.stderr)
            ipdb.post_mortem(tb)

def main():
    with contextlib.ExitStack() as context_stack:
        logging.basicConfig(filename="/tmp/stellarpunk.log", level=logging.INFO)
        #logging.basicConfig(stream=sys.stderr, level=logging.INFO)

        mgr = context_stack.enter_context(IPDBManager())
        ui = context_stack.enter_context(interface.Interface())

        ui.initialize()

        logging.info("generating universe...")
        generator = generate.UniverseGenerator()
        stellar_punk = generator.generate_universe()

        #logging.info("running simulation...")
        #stellar_punk.run()

        stellar_punk.production_chain.viz().render("production_chain", format="pdf")

        logging.info("done.")

if __name__ == "__main__":
    main()
