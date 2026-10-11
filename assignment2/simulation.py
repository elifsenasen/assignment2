# Standard library
from pathlib import Path
from typing import Literal

# Third-party libraries
import mujoco as mj
import numpy as np
import numpy.typing as npt
from mujoco import viewer

# Local libraries (ARIEL)
from ariel import console
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.prebuilt_robots import john_set
from ariel.simulation.environments import SimpleFlatWorld
from ariel.simulation.tasks.targeted_locomotion import distance_to_target
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.video_recorder import VideoRecorder

from controller import (
    angle_of_controller,
    genotype_length,
    genotype_to_weights,
    hidden_size,
    inputs_of_controller,
)

type ViewerTypes = Literal["launcher", "video", "simple", "frame", "no_control"]

# --- DATA SETUP --- #
SCRIPT_NAME = Path(__file__).stem
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(parents=True, exist_ok=True)

# --- EXPERIMENT CONSTANTS (keep FIXED across all compared runs) --- #
SPAWN_POS: list[float] = [0.0, 0.0, 0.1]
TARGET_POSITION: list[float] = [2.0, 0.0, 0.1]
SIM_DURATION: float = 15.0
NAN_FITNESS: float = 100.0


def build_world() -> SimpleFlatWorld:
    return SimpleFlatWorld()


def build_robot() -> CoreModule:
    return john_set.gecko()


def build_model() -> tuple[mj.MjModel, mj.MjData]:

    mj.set_mjcb_control(None)

    world = build_world()
    robot = build_robot()
    
    world.spawn(
        robot.spec,
        position=SPAWN_POS,
        correct_collision_with_floor=True,
    )
    model = world.spec.compile()
    data = mj.MjData(model)

    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    return model, data


_PROBLEM_SIZE: tuple[int, int, int] | None = None


def get_problem_size() -> tuple[int, int, int]:
    global _PROBLEM_SIZE
    if _PROBLEM_SIZE is None:
        model, data = build_model()
        input_size = inputs_of_controller(data, TARGET_POSITION).size
        output_size = model.nu
        n_genes = genotype_length(input_size, hidden_size, output_size)
        _PROBLEM_SIZE = (input_size, output_size, n_genes)
    return _PROBLEM_SIZE



def get_core_position(data: mj.MjData) -> npt.NDArray[np.float64]:
    return np.asarray(data.qpos[0:3]).copy()


def fitness_function(
    initial_position: npt.NDArray[np.float64],
    final_position: npt.NDArray[np.float64],
) -> float:
    
    final_position = np.asarray(final_position, dtype=float)
    if not np.all(np.isfinite(final_position[:2])):
        return NAN_FITNESS

    return distance_to_target(final_position, np.asarray(TARGET_POSITION, dtype=float))


def run_experiment(
    genotype: npt.ArrayLike,
    mode: ViewerTypes = "simple",
    verbose: bool = False,
) -> float:
    
    model, data = build_model()

    input_size, output_size = get_problem_size()[:2]
    genotype = np.asarray(genotype, dtype=float).ravel()
    W1, W2 = genotype_to_weights(genotype, input_size, hidden_size, output_size)

    def control_callback(m: mj.MjModel, d: mj.MjData) -> None:
        """Compute and apply actions; MuJoCo calls this every physics step."""
        inputs = inputs_of_controller(d, TARGET_POSITION)
        actions = angle_of_controller(inputs, W1, W2)
        d.ctrl[:] = np.nan_to_num(actions)

    initial_position = get_core_position(data)

    if mode != "no_control":
        mj.set_mjcb_control(control_callback)

    match mode:
        case "simple":
            simple_runner(model, data, duration=SIM_DURATION)
        case "launcher" | "no_control":
            viewer.launch(model=model, data=data)
        case "video":
            recorder = VideoRecorder(output_folder=str(DATA / "__videos__"))
            video_renderer(
                model,
                data,
                duration=SIM_DURATION,
                video_recorder=recorder,
            )
        case "frame":
            single_frame_renderer(model, data, steps=1, show=True)

   
    mj.set_mjcb_control(None)

    final_position = get_core_position(data)
    fitness = fitness_function(initial_position, final_position)

    if verbose:
        console.log(f"start  : {np.round(initial_position, 3)}")
        console.log(f"end    : {np.round(final_position, 3)}")
        console.log(f"target : {np.round(TARGET_POSITION, 3)}")
        console.log(f"fitness: {fitness:.4f}   (lower is better)")

    return fitness


def evaluate(genotype: npt.ArrayLike) -> float:
    return run_experiment(genotype, mode="simple", verbose=False)
