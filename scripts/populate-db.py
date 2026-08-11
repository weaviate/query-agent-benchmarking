import argparse
import query_agent_benchmarking

parser = argparse.ArgumentParser()
parser.add_argument("--no-poll", action="store_true", help="Submit to Engram without waiting for runs to complete")
args = parser.parse_args()

query_agent_benchmarking.populate_db(poll=not args.no_poll)
