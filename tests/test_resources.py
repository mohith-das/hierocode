from hierocode.runtime.recommendations import suggest_workers
from hierocode.models.schemas import ParallelizationConfig
from hierocode.providers.base import BaseProvider

class MockLocalProvider(BaseProvider):
    def healthcheck(self): return True
    def list_models(self): return []
    def generate(self, p, m, **kwargs): return ""
    def is_local(self): return True
    
class MockRemoteProvider(BaseProvider):
    def healthcheck(self): return True
    def list_models(self): return []
    def generate(self, p, m, **kwargs): return ""
    def is_local(self): return False

def test_suggest_workers_local():
    prov = MockLocalProvider("test", None)
    conf = ParallelizationConfig(max_local_workers=4)
    # Just checking it doesn't crash and returns ints
    assert isinstance(suggest_workers(prov, conf, "safe"), int)
    assert isinstance(suggest_workers(prov, conf, "balanced"), int)

def test_suggest_workers_remote():
    prov = MockRemoteProvider("test", None)
    conf = ParallelizationConfig(max_remote_workers=8)
    assert isinstance(suggest_workers(prov, conf, "safe"), int)
    assert isinstance(suggest_workers(prov, conf, "balanced"), int)
