"""Train recurrent IPPO or MAPPO on CoopSarSat with MAPX.

    python train_sarsat.py ippo  [hydra overrides...]
    python train_sarsat.py mappo [hydra overrides...]

The ``mapx/`` tree next to this file must have been copied over the MAPX checkout (see
README.md). Three things are swapped in before MAPX starts, none of which touches MAPX's
own files: the environment factory, the anchored actor / action head from
``mapx.networks.sarsat``, and a pass-through for ``AutoResetWrapper`` because the training
environment resets itself (``mapx.wrappers.sarsat_coop`` explains why).
"""

import importlib
import os
import sys

# SARSAT_HEAD=mixture selects the slot-mixture head instead of the slot-0 anchored one.
HEAD = os.environ.get("SARSAT_HEAD", "anchored")

# The ``mapx/`` overlay next to this script would shadow the installed package.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _here]


def main() -> None:
    system = sys.argv.pop(1)
    module = importlib.import_module(f"mapx.systems.ppo.anakin.rec_{system}")

    from mapx.networks.sarsat import AnchoredRecurrentActor, SlotMixtureActor
    from mapx.utils import make_env
    from mapx.wrappers import sarsat_coop
    from mapx.wrappers.sarsat_coop import PassthroughAutoReset, make_sarsat_coop_envs

    make_env.register_env_factory("SarSatCoop", make_sarsat_coop_envs)
    make_env.AutoResetWrapper = PassthroughAutoReset

    default_head = module.get_action_head

    def anchored_head(action_spec):
        head, kind = default_head(action_spec)
        name = "SlotMixtureHead" if HEAD == "mixture" else "AnchoredHybridHead"
        head = head | {
            "_target_": f"mapx.networks.sarsat.{name}",
            "slot_dim": sarsat_coop.LAST_SLOT_DIM,
        }
        return head, kind

    module.get_action_head = anchored_head
    module.Actor = SlotMixtureActor if HEAD == "mixture" else AnchoredRecurrentActor

    if not any(arg.startswith("env=") for arg in sys.argv):
        sys.argv.append("env=sarsat_coop")
    if not any(arg.startswith("network=") for arg in sys.argv):
        sys.argv.append("network=sarsat")
    module.hydra_entry_point()


if __name__ == "__main__":
    main()
