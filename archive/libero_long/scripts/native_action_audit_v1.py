"""Passive native OSC input inspection; never replaces or scales executed actions."""
import numpy as np


def native_action_contract(environment):
    # LIBERO ControlEnv deliberately does not forward action_spec. Fresh-reset
    # wrapper forwards .env to that same native robosuite environment.
    native = environment.env
    low, high = map(np.asarray, native.action_spec)
    assert low.shape == high.shape == (7,)
    assert len(native.robots) == 1
    robot = native.robots[0]
    controller = robot.controller
    assert controller.name == 'OSC_POSE'
    assert controller.impedance_mode == 'fixed' and controller.use_delta
    assert controller.control_dim == 6 and robot.gripper.dof == 1
    arm_low, arm_high = map(np.asarray, controller.control_limits)
    assert np.array_equal(low[:6], arm_low) and np.array_equal(high[:6], arm_high)
    return {
        'native': native, 'controller': controller, 'robot': robot,
        'input_low': low.copy(), 'input_high': high.copy(),
        'arm_low': arm_low.copy(), 'arm_high': arm_high.copy(),
        'output_low': np.asarray(controller.output_min).copy(),
        'output_high': np.asarray(controller.output_max).copy(),
    }


def passive_action_record(contract, action):
    action = np.asarray(action, dtype=np.float32)
    assert action.shape == (7,) and np.isfinite(action).all()
    low, high = contract['input_low'], contract['input_high']
    a_low, a_high = contract['arm_low'], contract['arm_high']
    clipped = np.clip(action[:6], a_low, a_high)
    out_low, out_high = contract['output_low'], contract['output_high']
    scale = np.abs(out_high-out_low) / np.abs(a_high-a_low)
    scaled = (clipped-(a_high+a_low)/2)*scale+(out_high+out_low)/2
    return {
        'controller_input': action.copy(),
        'input_bound_violations': int(np.count_nonzero((action < low) | (action > high))),
        'arm_clipped_components': int(np.count_nonzero((action[:6] < a_low) | (action[:6] > a_high))),
        'native_arm_clipped_input': clipped,
        'native_arm_scaled_delta': scaled,
        # Native PandaGripper uses sign(action), then clips cumulative actuator
        # state. This is not amplitude clipping of the seventh policy channel.
        'native_gripper_sign_command': float(np.sign(action[6])),
    }
