"""
Cache management using Redis
"""

import os
import json
import logging
from typing import Any, Optional
import redis.asyncio as redis

logger = logging.getLogger(__name__)

class CacheManager:
    """Redis cache manager"""
    
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis_client = None
        
    async def connect(self):
        """Connect to Redis"""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("Connected to Redis")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using in-memory cache.")
            self.redis_client = None
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.redis_client:
            return None
            
        try:
            value = await self.redis_client.get(key)
            if value:
                return json.loads(value)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in cache"""
        if not self.redis_client:
            return
            
        try:
            await self.redis_client.setex(key, ttl, json.dumps(value, default=str))
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    async def clear(self):
        """Clear all cache"""
        if self.redis_client:
            try:
                await self.redis_client.flushall()
            except Exception as e:
                logger.error(f"Cache clear error: {e}")
    
    async def get_stats(self):
        """Get cache statistics"""
        if not self.redis_client:
            return {"status": "disconnected"}
            
        try:
            info = await self.redis_client.info()
            return {
                "status": "connected",
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed")
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

# Global cache manager instance
cache_manager = CacheManager()