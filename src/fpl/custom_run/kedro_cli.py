import sys

if not sys.warnoptions:
    import os, warnings

    warnings.simplefilter("ignore")  # Change the filter in this process
    os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses

from itertools import chain
from pathlib import Path
from typing import Dict
import importlib
import urllib3

urllib3.disable_warnings()

import click
from kedro.framework.cli.cli import _init_plugins
from kedro.framework.cli.utils import (
    KedroCliError,
    env_option,
    split_string,
)
from itertools import chain
from pathlib import Path

import click

from kedro.framework.cli.utils import (
    KedroCliError,
    CONTEXT_SETTINGS,
)
from itertools import chain
from pathlib import Path
from typing import Sequence

import click
from kedro import __version__ as version
from kedro.framework.cli.catalog import catalog_cli
from kedro.framework.cli.hooks import CLIHooksManager
from kedro.framework.cli.jupyter import jupyter_cli
from kedro.framework.cli.pipeline import pipeline_cli
from kedro.framework.cli.project import project_group
from kedro.framework.cli.starters import create_cli
from kedro.framework.cli.utils import (
    CommandCollection,
    KedroCliError,
    load_entry_points,
    CONTEXT_SETTINGS,
)
from kedro.framework.startup import _is_project, bootstrap_project
import sys
from custom_runner import custom_kedro_run

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

# get our package onto the python path
PROJ_PATH = Path(__file__).resolve().parent

ENV_ARG_HELP = """Run the pipeline in a configured environment. If not specified,
pipeline will run using environment `local`."""
FROM_INPUTS_HELP = (
    """A list of dataset names which should be used as a starting point."""
)
TO_OUTPUTS_HELP = """A list of dataset names which should be used as an end point."""
FROM_NODES_HELP = """A list of node names which should be used as a starting point."""
TO_NODES_HELP = """A list of node names which should be used as an end point."""
NODE_ARG_HELP = """Run only nodes with specified names."""
RUNNER_ARG_HELP = """Specify a runner that you want to run the pipeline with.
Available runners: `SequentialRunner`, `ParallelRunner` and `ThreadRunner`.
This option cannot be used together with --parallel."""
PARALLEL_ARG_HELP = """Run the pipeline using the `ParallelRunner`.
If not specified, use the `SequentialRunner`. This flag cannot be used together
with --runner."""
ASYNC_ARG_HELP = """Load and save node inputs and outputs asynchronously
with threads. If not specified, load and save datasets synchronously."""
TAG_ARG_HELP = """Construct the pipeline using only nodes which have this tag
attached. Option can be used multiple times, what results in a
pipeline constructed from nodes having any of those tags."""
LOAD_VERSION_HELP = """Specify a particular dataset version (timestamp) for loading."""
CONFIG_FILE_HELP = """Specify a YAML configuration file to load the run
command arguments from. If command line arguments are provided, they will
override the loaded ones."""
PIPELINE_ARG_HELP = """Name of the modular pipeline to run.
If not set, the project pipeline is run by default."""
PARAMS_ARG_HELP = """Specify extra parameters that you want to pass
to the context initializer. Items must be separated by comma, keys - by colon,
example: param1:value1,param2:value2. Each parameter is split by the first comma,
so parameter values are allowed to contain colons, parameter keys are not."""


@click.group(context_settings=CONTEXT_SETTINGS, name="Kedro")
@click.version_option(version, "--version", "-V", help="Show version and exit")
def cli():
    pass


class CustomKedroCLI(CommandCollection):
    def __init__(self, project_path: Path):
        self._metadata = None  # running in package mode
        if _is_project(project_path):
            self._metadata = bootstrap_project(project_path)
        self._cli_hook_manager = CLIHooksManager()

        groups = [
            ("Global commands", self.global_groups),
            ("Project specific commands", self.project_groups),
        ]

        self.groups = [
            (title, self._merge_same_name_collections(cli_list))
            for title, cli_list in groups
        ]
        sources = list(chain.from_iterable(cli_list for _, cli_list in self.groups))

        help_texts = [
            cli.help
            for cli_collection in sources
            for cli in cli_collection.sources
            if cli.help
        ]
        self._dedupe_commands(sources)

        click.CommandCollection.__init__(
            self,
            name="custom_kedro_cli",
            sources=sources,
            help="\n\n".join(help_texts),
            context_settings=CONTEXT_SETTINGS,
        )
        self.params = sources[0].params
        self.callback = sources[0].callback

    def main(
        self,
        args=None,
        prog_name=None,
        complete_var=None,
        standalone_mode=True,
        **extra,
    ):
        if self._metadata:
            extra.update(obj=self._metadata)

        # This is how click's internals parse sys.argv, which include the command,
        # subcommand, arguments and options. click doesn't store this information anywhere
        # so we have to re-do it.
        # https://github.com/pallets/click/blob/master/src/click/core.py#L942-L945
        args = sys.argv[1:] if args is None else list(args)
        self._cli_hook_manager.hook.before_command_run(
            project_metadata=self._metadata, command_args=args
        )

        super().main(
            args=args,
            prog_name=prog_name,
            complete_var=complete_var,
            standalone_mode=standalone_mode,
            **extra,
        )

    @property
    def global_groups(self) -> Sequence[click.MultiCommand]:
        """Property which loads all global command groups from plugins and
        combines them with the built-in ones (eventually overriding the
        built-in ones if they are redefined by plugins).
        """
        return [cli, create_cli, *load_entry_points("global")]

    @property
    def project_groups(self) -> Sequence[click.MultiCommand]:
        """Property which loads all project command groups from the
        project and the plugins, then combines them with the built-in ones.
        Built-in commands can be overridden by plugins, which can be
        overridden by the project's cli.py.
        """
        if not self._metadata:
            return []

        built_in = [catalog_cli, jupyter_cli, pipeline_cli, project_group]

        plugins = load_entry_points("project")

        project_cli = importlib.import_module(
            f"{self._metadata.package_name}.custom_run.kedro_cli"
        )
        user_defined = project_cli.cli
        return [*built_in, *plugins, user_defined]


def _config_file_callback(ctx, param, value):  # pylint: disable=unused-argument
    """Config file callback, that replaces command line options with config file
    values. If command line options are passed, they override config file values.
    """
    # for performance reasons
    import anyconfig  # pylint: disable=import-outside-toplevel

    ctx.default_map = ctx.default_map or {}
    # section = ctx.info_name

    if value:
        config = anyconfig.load(value)
        config["run"].pop("hyperopt", None)
        config["run"].pop("experiment", None)
        ctx.default_map.update(config["run"])

        ctx.params["hyperopt"] = config.get("hyperopt")
        ctx.params["experiment"] = config.get("experiment")

    return value


def _reformat_load_versions(  # pylint: disable=unused-argument
    ctx, param, value
) -> Dict[str, str]:
    """Reformat data structure from tuple to dictionary for `load-version`.
    E.g ('dataset1:time1', 'dataset2:time2') -> {"dataset1": "time1", "dataset2": "time2"}.
    """
    load_versions_dict = {}

    for load_version in value:
        load_version_list = load_version.split(":", 1)
        if len(load_version_list) != 2:
            raise KedroCliError(
                f"Expected the form of `load_version` to be "
                f"`dataset_name:YYYY-MM-DDThh.mm.ss.sssZ`,"
                f"found {load_version} instead"
            )
        load_versions_dict[load_version_list[0]] = load_version_list[1]

    return load_versions_dict


def _split_params(ctx, param, value):
    if isinstance(value, dict):
        return value
    result = {}
    for item in split_string(ctx, param, value):
        item = item.split(":", 1)
        if len(item) != 2:
            ctx.fail(
                f"Invalid format of `{param.name}` option: Item `{item[0]}` must contain "
                f"a key and a value separated by `:`."
            )
        key = item[0].strip()
        if not key:
            ctx.fail(
                f"Invalid format of `{param.name}` option: Parameter key "
                f"cannot be an empty string."
            )
        value = item[1].strip()
        result[key] = _try_convert_to_numeric(value)
    return result


def _try_convert_to_numeric(value):
    try:
        value = float(value)
    except ValueError:
        return value
    return int(value) if value.is_integer() else value


@click.group(context_settings=CONTEXT_SETTINGS, name=__file__)
def cli():
    """Command line tools for manipulating a Kedro project."""


@cli.command()
@click.option(
    "--from-inputs", type=str, default="", help=FROM_INPUTS_HELP, callback=split_string
)
@click.option(
    "--to-outputs", type=str, default="", help=TO_OUTPUTS_HELP, callback=split_string
)
@click.option(
    "--from-nodes", type=str, default="", help=FROM_NODES_HELP, callback=split_string
)
@click.option(
    "--to-nodes", type=str, default="", help=TO_NODES_HELP, callback=split_string
)
@click.option("--node", "-n", "node_names", type=str, multiple=True, help=NODE_ARG_HELP)
@click.option(
    "--runner", "-r", type=str, default=None, multiple=False, help=RUNNER_ARG_HELP
)
@click.option("--parallel", "-p", is_flag=True, multiple=False, help=PARALLEL_ARG_HELP)
@click.option("--async", "is_async", is_flag=True, multiple=False, help=ASYNC_ARG_HELP)
@env_option
@click.option("--tag", "-t", type=str, multiple=True, help=TAG_ARG_HELP)
@click.option(
    "--load-version",
    "-lv",
    type=str,
    multiple=True,
    help=LOAD_VERSION_HELP,
    callback=_reformat_load_versions,
)
@click.option("--pipeline", type=str, default=None, help=PIPELINE_ARG_HELP)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    help=CONFIG_FILE_HELP,
    callback=_config_file_callback,
    expose_value=False,
)
@click.option(
    "--params", type=str, default="", help=PARAMS_ARG_HELP, callback=_split_params
)
@click.option(
    "--no_upload",
    default=False,
    is_flag=True,
    help=PARAMS_ARG_HELP,
)
@click.option(
    "--no_download",
    default=False,
    is_flag=True,
    help=PARAMS_ARG_HELP,
)
def run(
    env,
    params,
    parallel,
    runner,
    is_async,
    node_names,
    to_nodes,
    from_nodes,
    from_inputs,
    to_outputs,
    load_version,
    pipeline,
    tag,
    hyperopt=None,
    experiment=None,
    no_upload: bool = False,
    no_download: bool = False,
):
    """Kedro pipeline custom run."""
    if env is None:
        env = "local"

    custom_kedro_run(
        env=env,
        params=params,
        project_path=Path.cwd(),
        run={
            "parallel": parallel,
            "runner": runner,
            "is_async": is_async,
            "node_names": node_names,
            "to_nodes": to_nodes,
            "from_nodes": from_nodes,
            "from_inputs": from_inputs,
            "to_outputs": to_outputs,
            "load_version": load_version,
            "pipeline": pipeline,
            "tag": tag,
        },
    )


def main():  # pragma: no cover
    _init_plugins()
    cli_collection = CustomKedroCLI(project_path=Path.cwd())
    cli_collection()


if __name__ == "__main__":
    main()
