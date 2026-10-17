"""Reconstruct the unchanged native environment to restore model/XML hidden state."""
class NativeFreshResetEnv:
    def __init__(self, constructor=None, **kwargs):
        if constructor is None:
            from libero.libero.envs import OffScreenRenderEnv
            constructor = OffScreenRenderEnv
        self._constructor = constructor
        self._kwargs = dict(kwargs)
        self._environment = None
        self._seed = None

    def seed(self, seed):
        self._seed = seed

    def reset(self):
        if self._environment is not None:
            self._environment.close()
        self._environment = self._constructor(**self._kwargs)
        assert self._seed is not None
        self._environment.seed(self._seed)
        return self._environment.reset()

    def close(self):
        if self._environment is not None:
            self._environment.close()
            self._environment = None

    def __getattr__(self, key):
        assert self._environment is not None
        return getattr(self._environment, key)
