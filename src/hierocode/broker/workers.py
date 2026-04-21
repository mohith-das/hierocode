import concurrent.futures
from typing import List, Callable

def run_draft_workers(
    worker_count: int, 
    task_func: Callable[[], str]
) -> List[str]:
    """
    Runs a uniform drafting task multiple times in parallel to get candidates.
    Returns a list of drafted outputs.
    """
    candidates = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(task_func) for _ in range(worker_count)]
        for future in concurrent.futures.as_completed(futures):
            try:
                res = future.result()
                if res:
                    candidates.append(res)
            except Exception:
                pass # In v0.1 we can just ignore failing drafts if we get at least one
    return candidates
