"""
Hydra entry point for the activitygraphs experiment pipeline.

This module is the command-line entry point for running the full comparison
experiment.  It:
  1. Registers the ``Config`` dataclass with Hydra's ConfigStore so that Hydra
     knows the expected configuration structure and can validate YAML files.
  2. Defines the ``main()`` function, decorated with ``@hydra.main``, which
     Hydra calls after loading and merging all configuration files.
  3. Passes the resolved ``Config`` object to ``comparison_experiment()`` in
     ``activitygraphs.run``, which does all the actual work.

Usage (from the repository root):
    python -m activitygraphs.main data=toronto train.epochs=100
    python -m activitygraphs.main data=geneva train.fast_dev_run=true
"""

import hydra
from hydra.core.config_store import ConfigStore

from activitygraphs.config import Config
from activitygraphs.run import comparison_experiment

# Retrieve the global Hydra ConfigStore singleton.
# Registering the Config dataclass here tells Hydra the expected schema so it
# can validate the YAML files against the dataclass field types and names.
cs = ConfigStore.instance()
# Store the Config schema under the name "geneva_config".
# The actual config selected at runtime is still controlled by conf/config.yaml
# and the ``data=`` command-line override.
cs.store(name="geneva_config", node=Config)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: Config):
    """
    Description: Hydra-decorated entry-point function.  Hydra calls this function
    automatically after loading and merging all YAML configuration files.  The
    resolved configuration is passed in as ``cfg``.

    This function is intentionally thin: it simply delegates all work to
    ``comparison_experiment(cfg)`` so that the actual experiment logic is easy
    to test and reuse independently of Hydra.

    Input:
      - cfg (Config): fully populated OmegaConf Config object built by Hydra
                      from conf/config.yaml plus any command-line overrides.

    Output:
      - (None): all results are written to disk by comparison_experiment().
    """
    # Delegate all experiment logic to comparison_experiment in activitygraphs.run.
    comparison_experiment(cfg)


if __name__ == "__main__":
    # When this module is executed directly (``python -m activitygraphs.main``),
    # call main() which Hydra will intercept to parse CLI arguments and load configs.
    main()
