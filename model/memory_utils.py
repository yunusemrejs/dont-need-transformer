from .common_imports import *
from .config import SiegelModelConfig
import weakref

class MemoryManager:
    """Shared memory optimization utilities with advanced caching."""

    def __init__(self, config: SiegelModelConfig):
        self.config = config
        self._init_caches()
        self._init_metrics()

    def _init_caches(self):
        """Initialize cache structures."""
        self.pattern_cache = LRUCache(self.config.cache_size)
        self.attention_cache = LRUCache(self.config.cache_size)
        self.tensor_cache = WeakTensorCache()

    def _init_metrics(self):
        """Initialize performance metrics."""
        self.metrics = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_saved': 0
        }

    @torch.jit.script_method
    def _compute_cache_key(self, args: Tuple) -> str:
        """Compute cache key for given arguments."""
        key_parts = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                # Use shape and device as key components
                key_parts.extend([
                    str(arg.shape),
                    str(arg.device),
                    str(hash(arg.data_ptr()))
                ])
            else:
                key_parts.append(str(arg))
        return "_".join(key_parts)

    def efficient_attention(
        self,
        func: Callable,
        *args,
        use_checkpoint: bool = True,
        **kwargs
    ) -> Any:
        """Execute attention with memory optimization and caching."""
        if self.training:
            # Don't use cache during training
            return self._execute_attention(func, args, use_checkpoint, kwargs)

        # Compute cache key
        cache_key = self._compute_cache_key(args)

        # Try to get from cache
        if cache_key in self.attention_cache:
            self.metrics['hits'] += 1
            return self.attention_cache[cache_key]

        # Execute and cache result
        self.metrics['misses'] += 1
        result = self._execute_attention(func, args, use_checkpoint, kwargs)
        self.attention_cache[cache_key] = result

        return result

    def _execute_attention(
        self,
        func: Callable,
        args: Tuple,
        use_checkpoint: bool,
        kwargs: Dict
    ) -> Any:
        """Execute attention function with checkpointing if needed."""
        if use_checkpoint and self.config.use_gradient_checkpointing:
            return checkpoint(func, *args)
        return func(*args, **kwargs)

    def get_metrics(self) -> Dict[str, float]:
        """Get cache performance metrics."""
        total = self.metrics['hits'] + self.metrics['misses']
        if total == 0:
            return {k: 0.0 for k in self.metrics}

        return {
            'hit_rate': self.metrics['hits'] / total,
            'miss_rate': self.metrics['misses'] / total,
            'eviction_rate': self.metrics['evictions'] / total,
            'memory_saved_mb': self.metrics['memory_saved'] / (1024 * 1024)
        }

    def clear_cache(self, clear_metrics: bool = False):
        """Clear cached computations and optionally reset metrics."""
        self.pattern_cache.clear()
        self.attention_cache.clear()
        self.tensor_cache.clear()
        torch.cuda.empty_cache()

        if clear_metrics:
            self._init_metrics()

class LRUCache:
    """Least Recently Used Cache implementation."""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.usage = []

    def __getitem__(self, key: str) -> Any:
        if key in self.cache:
            # Move to most recently used
            self.usage.remove(key)
            self.usage.append(key)
            return self.cache[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any):
        if len(self.cache) >= self.capacity:
            # Remove least recently used
            lru_key = self.usage.pop(0)
            del self.cache[lru_key]

        self.cache[key] = value
        self.usage.append(key)

    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.usage.clear()

class WeakTensorCache:
    """Cache for tensor operations using weak references."""
    def __init__(self):
        self.cache = weakref.WeakKeyDictionary()

    def __getitem__(self, tensor: torch.Tensor) -> Any:
        return self.cache[tensor]

    def __setitem__(self, tensor: torch.Tensor, value: Any):
        self.cache[tensor] = value

    def clear(self):
        """Clear the cache."""
        self.cache.clear()
