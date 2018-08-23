"""
An interface for asynchronous vectorized environments.
"""

from multiprocessing import Pipe, Array, Process
import numpy as np
from . import VecEnv, CloudpickleWrapper
import ctypes

from .util import dict_to_obs, obs_space_info, obs_to_dict
from unityagents.curriculum import Curriculum
from unityagents.brain import BrainInfo, BrainParameters, AllBrainInfo

import gym
_NP_TO_CT = {np.float32: ctypes.c_float,
             np.int32: ctypes.c_int32,
             np.int8: ctypes.c_int8,
             np.uint8: ctypes.c_char,
             np.bool: ctypes.c_bool}


class ShmemVecEnv(VecEnv):
    """
    An AsyncEnv that uses multiprocessing to run multiple
    environments in parallel.
    """

    def __init__(self, env_fns, spaces=None,worker_id=0,
                 base_port=5005, curriculum=None,
                 seed=0, docker_training=False, no_graphics=False,brain_name="CustomBrain"):
        """
        If you don't specify observation_space, we'll have to create a dummy
        environment to get it.
        """
        if spaces:
            observation_space, action_space = spaces
        else:
            print('Creating dummy env object to get spaces')
            dummy = env_fns[0](0)
            observation_space, action_space = dummy.observation_space, dummy.action_space
            dummy.close()
            del dummy
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)
        self.obs_keys, self.obs_shapes, self.obs_dtypes = obs_space_info(observation_space)
        self.obs_bufs = [
            {k: Array(_NP_TO_CT[self.obs_dtypes[k].type], int(np.prod(self.obs_shapes[k]))) for k in self.obs_keys}
            for _ in env_fns]
        self.parent_pipes = []
        self.procs = []
        self.agents = []
        for idx,(env_fn, obs_buf) in enumerate(zip(env_fns, self.obs_bufs)):
            wrapped_fn = CloudpickleWrapper(env_fn)
            parent_pipe, child_pipe = Pipe()
            proc = Process(target=_subproc_worker,
                           args=(child_pipe, parent_pipe, wrapped_fn, obs_buf, self.obs_shapes, self.obs_dtypes, self.obs_keys,idx))
            proc.daemon = True
            self.procs.append(proc)
            self.parent_pipes.append(parent_pipe)
            proc.start()
            child_pipe.close()
            self.agents += [str(idx)]
        self._n_agents = len(self.agents)
        self.waiting_step = False
        class Viewer(object):
            def __init__(self):
                return
            def close(self):
                return
        self.viewer = Viewer()
        self._brains = {}
        brain_name = brain_name if  isinstance(action_space, gym.spaces.Discrete) else "Continuous"+brain_name
        self._brain_names = [brain_name]
        self._external_brain_names = []
        self.num_actions=action_space.shape[0]
        resolution = None
        if isinstance(observation_space, gym.spaces.Box):
            assert len(observation_space.shape) == 3, "Only H,W,C continuous obeservations possible"
            resolution = [{
                "height": observation_space.shape[0],
                "width": observation_space.shape[1],
                "blackAndWhite": observation_space.shape[2]
            }]

        self._brains[brain_name] = \
            BrainParameters(brain_name, {
                "vectorObservationSize": 0 if isinstance(observation_space, gym.spaces.Box) else observation_space.n,
                "numStackedVectorObservations": len(env_fns) if isinstance(observation_space, gym.spaces.Discrete) else 0,
                "cameraResolutions": resolution,
                "vectorActionSize": action_space.n if isinstance(action_space, gym.spaces.Discrete) else action_space.shape[0],
                "vectorActionDescriptions": ["", ""],
                "vectorActionSpaceType": 0 if isinstance(action_space, gym.spaces.Discrete) else 1,
                "vectorObservationSpaceType": 0 if isinstance(observation_space,gym.spaces.Discrete) else 1
            })
        self._external_brain_names += [brain_name]
        self._num_brains = len(self._brain_names)
        self._num_external_brains = len(self._external_brain_names)
        self._curriculum = Curriculum(curriculum, None)
        self._global_done = None
        self.viewer = None

    def reset(self,train_mode=None):
        self.train_mode = train_mode
        if self.waiting_step:
            print('Called reset() while waiting for the step to complete')
            self.step_wait()
        for pipe in self.parent_pipes:
            pipe.send(('reset', self.train_mode))

        states = self._decode_obses([pipe.recv() for pipe in self.parent_pipes])
        state = {self._brain_names[0]:BrainInfo([states],[],[], max_reached=[False]*self._n_agents,local_done=[False]*self._n_agents,agents= self.agents,vector_action=np.array([[0]*self.num_actions]*self._n_agents),memory=np.array([[]]*self._n_agents))}
        return state

    def step(self, vector_action=None, memory=None, text_action=None):
        memory = memory[self._brain_names[0]]
        vector_action = vector_action[self._brain_names[0]]

        self.step_async(vector_action)
        results = self.step_wait()

        state = {self._brain_names[0]: BrainInfo([results[0]], [], [], local_done=results[2], reward=results[1], agents=self.agents,
                                                vector_action=vector_action,
                                                memory=memory, max_reached=results[2])}
        self._global_done = any(results[2])
        return state

    def step_async(self, actions):
        assert len(actions) == len(self.parent_pipes)
        for pipe, act in zip(self.parent_pipes, actions):
            pipe.send(('step', act))

    def step_wait(self):
        outs = [pipe.recv() for pipe in self.parent_pipes]
        obs, rews, dones, infos = zip(*outs)
        return self._decode_obses(obs), np.array(rews), np.array(dones), infos

    def close(self):
        if self.waiting_step:
            self.step_wait()
        for pipe in self.parent_pipes:
            pipe.send(('close', None))
        for pipe in self.parent_pipes:
            pipe.recv()
            pipe.close()
        for proc in self.procs:
            proc.join()
        if self.viewer is not None:
            self.viewer.close()


    def _decode_obses(self, obs):
        result = {}
        for k in self.obs_keys:

            bufs = [b[k] for b in self.obs_bufs]
            o = [np.frombuffer(b.get_obj(), dtype=self.obs_dtypes[k]).reshape(self.obs_shapes[k]) for b in bufs]
            result[k] = np.array(o)
        return dict_to_obs(result)

    @property
    def curriculum(self):
        return self._curriculum

    @property
    def logfile_path(self):
        return self._log_path

    @property
    def brains(self):
        return self._brains

    @property
    def global_done(self):
        return self._global_done

    @property
    def academy_name(self):
        return self._academy_name

    @property
    def number_brains(self):
        return self._num_brains

    @property
    def number_external_brains(self):
        return self._num_external_brains

    @property
    def brain_names(self):
        return self._brain_names

    @property
    def external_brain_names(self):
        return self._external_brain_names


def _subproc_worker(pipe, parent_pipe, env_fn_wrapper, obs_bufs, obs_shapes, obs_dtypes, keys,idx):
    """
    Control a single environment instance using IPC and
    shared memory.
    """
    def _write_obs(maybe_dict_obs):
        flatdict = obs_to_dict(maybe_dict_obs)
        for k in keys:
            dst = obs_bufs[k].get_obj()
            dst_np = np.frombuffer(dst, dtype=obs_dtypes[k]).reshape(obs_shapes[k])  # pylint: disable=W0212
            np.copyto(dst_np, flatdict[k])

    env = env_fn_wrapper.x(idx)
    parent_pipe.close()
    try:
        while True:
            cmd, data = pipe.recv()
            if cmd == 'reset':
                pipe.send(_write_obs(env.reset(data)))
            elif cmd == 'step':
                obs, reward, done, info = env.step(data)
                if done:
                    obs = env.reset()
                pipe.send((_write_obs(obs), reward, done, info))
            elif cmd == 'render':
                pipe.send(env.render(mode='rgb_array'))
            elif cmd == 'close':
                pipe.send(None)
                break
            else:
                raise RuntimeError('Got unrecognized cmd %s' % cmd)
    except KeyboardInterrupt:
        print('ShmemVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()
