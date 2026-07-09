
# Global MemoryStore singleton
_memory_store = None

def set_memory_store(store):
    global _memory_store
    _memory_store = store

def get_memory_store():
    return _memory_store