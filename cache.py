
import redis
import os

redis_client = redis.Redis(host="redis", port=6379, decode_responses=True)

def cache_response(query):
    return redis_client.get(query)

def set_cache(query, value):
    redis_client.set(query, value, ex=3600)
