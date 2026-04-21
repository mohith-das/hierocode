from hierocode.runtime.resources import get_cpu_count, get_total_ram_gb, get_available_ram_gb
from hierocode.models.schemas import ParallelizationConfig
from hierocode.providers.base import BaseProvider

def suggest_workers(provider: BaseProvider, config: ParallelizationConfig, strategy: str = "balanced") -> int:
    """
    Provide a heuristic worker count recommendation based on the system resources
    and whether the provider is remote or local.
    """
    cpus = get_cpu_count()
    ram_gb = get_available_ram_gb()
    is_local = provider.is_local()
    
    if is_local:
        # Local inference is heavy on RAM and CPU.
        # If we have low ram, drop heavily.
        base_workers = max(1, cpus // 2)
        if ram_gb < 8.0:
            base_workers = 1
        elif ram_gb > 32.0:
            base_workers = max(2, cpus)
            
        # Clamp to configured maximum
        base_workers = min(base_workers, config.max_local_workers)
    else:
        # Remote inference is mostly IO bound but we don't want to rate limit ourselves hard.
        base_workers = cpus * 2
        base_workers = min(base_workers, config.max_remote_workers)
        
    # Apply strategy
    if strategy == "safe":
        return max(1, base_workers // 2)
    elif strategy == "aggressive":
        return max(1, int(base_workers * 1.5))
    else:
        # balanced
        return max(1, base_workers)
