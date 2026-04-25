import pytest
from env import WildfireEnv


@pytest.fixture
def fresh_env():
    env = WildfireEnv()
    yield env
