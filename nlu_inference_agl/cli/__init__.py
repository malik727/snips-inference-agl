import argparse


class Formatter(argparse.ArgumentDefaultsHelpFormatter):
    def __init__(self, prog):
        super(Formatter, self).__init__(prog, max_help_position=35, width=150)


def get_arg_parser():
    from nlu_inference_agl.cli.inference import add_parse_parser
    from nlu_inference_agl.cli.versions import (
        add_version_parser, add_model_version_parser)

    arg_parser = argparse.ArgumentParser(
        description="Snips NLU command line interface",
        prog="python -m snips_nlu", formatter_class=Formatter)
    arg_parser.add_argument("-v", "--version", action="store_true",
                            help="Print package version")
    subparsers = arg_parser.add_subparsers(
        title="available commands", metavar="command [options ...]")
    add_parse_parser(subparsers, formatter_class=Formatter)
    add_version_parser(subparsers, formatter_class=Formatter)
    add_model_version_parser(subparsers, formatter_class=Formatter)
    return arg_parser


def main():
    from nlu_inference_agl.__about__ import __version__

    arg_parser = get_arg_parser()
    args = arg_parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    elif "version" in args:
        print(__version__)
    else:
        arg_parser.print_help()
        exit(1)
