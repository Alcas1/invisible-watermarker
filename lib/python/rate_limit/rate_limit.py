from datetime import timedelta
from redis import Redis
import os

r = Redis.from_url(os.environ.get("REDIS_URL"))


def request_is_limited(r: Redis, key: str, limit: int, period: timedelta):
  # gcra algorithm
  period_in_seconds = int(period.total_seconds())
  t = r.time()[0]
  separation = round(period_in_seconds / limit)
  r.setnx(key, 0)
  tat = max(int(r.get(key)), t)
  if tat - t <= period_in_seconds - separation:
    new_tat = max(tat, t) + separation
    r.set(key, new_tat)
    return False
  return True
