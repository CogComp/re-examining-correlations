import argparse
import contextlib
import math

from sacrerouge.io import JsonlReader, JsonlWriter


def main(args):
    # Read all of the instances in so we can maintain the order across the files
    instances = JsonlReader(args.input_jsonl).read()

    num_instances = len(instances)
    num_instances_per_chunk = int(math.ceil(num_instances / args.num_batches))

    with contextlib.ExitStack() as stack:
        writers = [stack.enter_context(JsonlWriter(f'{args.output_dir}/{i}.jsonl.gz')) for i in range(args.num_batches)]
        for i, writer in zip(range(0, len(instances), num_instances_per_chunk), writers):
            for instance in instances[i:i + num_instances_per_chunk]:
                writer.write(instance)


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument("--input-jsonl", required=True)
    argp.add_argument("--num-batches", required=True, type=int)
    argp.add_argument("--output-dir", required=True)
    args = argp.parse_args()
    main(args)