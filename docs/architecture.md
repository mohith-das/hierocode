# Architecture

Hierocode uses a decoupled architecture allowing multiple logic modules to operate independently and scalably.

## Components

1. **CLI Layer**: Defines commands to parse input arguments and pass configurations deep into the business logic.
2. **Provider Abstract Layer**: Unified adapter interface over varied backend providers (Ollama, OpenAI standard endpoints, LM Studio, etc.).
3. **Broker**: Orchestrates complex flows:
    - `planner.py`: Divides a large user task into file-specific goals.
    - `router.py`: Selects the appropriate model given task characteristics.
    - `workers.py`: Dispatches localized tasks to smaller models asynchronously using standard Python thread pools.
    - `escalation.py`: Determines if context exceeds local model capacity or if error counts trigger fallback to a more capable model.
4. **Runtime Resource Monitor**: Examines the immediate hardware constraints to limit local AI processes.
